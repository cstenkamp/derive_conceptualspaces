import warnings
from collections import Counter

from plotly.serializers import np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.jsonloadstore import Struct
from derive_conceptualspace.util.mpl_tools import show_hist

from misc_util.pretty_print import pretty_print as print

flatten = lambda l: [item for sublist in l for item in sublist]


class DocTermMatrix():

    @staticmethod
    def fromstruct(struct):
        assert not struct["includes_pseudodocs"], "TODO"
        return DocTermMatrix(dtm=struct["dtm"], all_terms={int(k): v for k, v in struct["all_terms"].items()}, quant_name=struct["quant_name"])

    @staticmethod
    def from_descriptions(descriptions, **kwargs):
        return DocTermMatrix.from_vocab_descriptions(descriptions.all_words(), descriptions, **kwargs)

    @staticmethod
    def from_vocab_descriptions(vocab, descriptions, min_df=1, verbose=False):
        if hasattr(descriptions, "recover_settings"):
            warnings.warn("Are you sure you don't want to use descriptions.generate_DocTermMatrix instead?")
        assert descriptions.proc_min_df <= min_df
        all_terms = {n: elem for n, elem in enumerate(vocab)}
        reverse_term_dict = {v: k for k,v in all_terms.items()}
        dtm = []
        for desc in tqdm(descriptions._descriptions, desc="Loading Bag-Of-Words to DocTermMatrix.."):
            dtm.append([[reverse_term_dict[k], v] for k,v in desc.bow().items()])
        if min_df > 1:
            return DocTermMatrix.filter(dtm, min_df, use_n_docs_count=True, verbose=verbose, descriptions=descriptions, all_terms=all_terms) #TODO make use_n_docs_count an arg
        return DocTermMatrix(dtm, all_terms, quant_name="count")

    def __init__(self, dtm, all_terms, quant_name, verbose=False, **kwargs): #TODO overhaul 16.01.2022: much to delete here?!
        self.includes_pseudodocs = False
        self.dtm = dtm
        self.all_terms = {n: elem for n, elem in enumerate(all_terms)} if isinstance(all_terms, list) else all_terms
        self.quant_name = quant_name
        # if len(args) == 1 and isinstance(args[0], dict):
        #     assert "doc_term_matrix" in args[0] and "all_terms" in args[0]
        #     assert not kwargs
        #     self.dtm = args[0]["doc_term_matrix"]
        #     self.all_terms = args[0]["all_terms"]
        #     if isinstance(next(iter(self.all_terms.keys())), str):
        #         self.all_terms = {int(k): v for k, v in self.all_terms.items()}
        #     #TODO store meta-info
        #     #TODO assert dass len(self.dtm) == len(mds_obj.names)
        # # elif "all_terms" in kwargs and "descriptions" in kwargs:
        # #     # assert hasattr(kwargs["descriptions"][0], "bow")
        # #     if isinstance(kwargs["all_terms"], dict):
        # #         self.all_terms = kwargs["all_terms"]
        # #     else:
        # #         self.all_terms = {n: elem for n, elem in enumerate(kwargs["all_terms"])}
        # #     self.dtm = []
        # #     for desc in kwargs["descriptions"]:
        # #         self.dtm.append([[self.reverse_term_dict[k], v] for k,v in desc.bow().items()])
        # elif "all_phrases" in kwargs and "descriptions" in kwargs and "dtm" in kwargs:
        #     if isinstance(kwargs["all_phrases"], dict):
        #         self.all_terms = kwargs["all_phrases"]
        #     else:
        #         self.all_terms = {n: elem for n, elem in enumerate(kwargs["all_phrases"])}
        #     self.dtm = kwargs["dtm"]
        #     self.descriptions = kwargs["descriptions"]
        # else:
        #     assert False
        assert set(self.all_terms) == set(flatten([[elem[0] for elem in row] for row in self.dtm]))
        print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")
        if verbose:
            self.show_info()

    def json_serialize(self):
        return Struct(**{k:v for k,v in self.__dict__.items() if not k.startswith("_") and k not in ["csr_matrix", "doc_freqs", "reverse_term_dict"]})

    n_docs = property(lambda self: len(self.dtm))

    def show_info(self, descriptions=None):
        occurs_in = [set(j[0] for j in i) if i else [] for i in self.dtm]
        num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(self.all_terms)), desc="Counting Occurences [verbose]")]
        show_hist(num_occurences, "Docs per Keyword", xlabel="# Documents the Keyword appears in", ylabel="Count (log scale)", cutoff_percentile=98, log=True)
        above_threshold = len([i for i in num_occurences if i>= get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)])
        sorted_canditerms = sorted([[ind, elem] for ind, elem in enumerate(num_occurences)], key=lambda x:x[1], reverse=True)
        print(f"Found {len(self.all_terms)} candidate Terms, {above_threshold} ({round(above_threshold/len(self.all_terms)*100)}%) of which occur in at least {get_setting('CANDIDATE_MIN_TERM_COUNT', silent=True)} descriptions.")
        print("The 25 terms that occur in the most descriptions (incl the #descriptions they occur in):",
              ", ".join([f"{self.all_terms[ind]} ({occs})" for ind, occs in sorted_canditerms[:25]]))
        if descriptions is not None:
            max_ind = np.unravel_index(self.as_csr().argmax(), self.as_csr().shape)
            print(f"Max value: Term *b*{self.all_terms[max_ind[0]]}*b* has value *b*{dict(self.dtm[max_ind[1]])[max_ind[0]]:.3f}*b* for doc *b*{descriptions._descriptions[max_ind[1]].title}*b*")

    #num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(dtm.all_terms)))]

    @property
    def doc_freqs(self):
        """the number of documents containing a word, for all words"""
        if not hasattr(self, "_doc_freqs"):
            occurences = [set(i[0] for i in doc) for doc in self.dtm]
            self._doc_freqs = {term: sum(term in doc for doc in occurences) for term in tqdm(list(self.all_terms.keys()), desc="Calculating Doc-Frequencies")}
            print("Most frequent term:", self.all_terms[max(self._doc_freqs.items(), key=lambda x:x[1])[0]])
        return self._doc_freqs

    def terms_per_doc(self):
        if not hasattr(self, "_terms_per_doc"):
            self._terms_per_doc = [[self.all_terms[j[0]] for j in i] for i in self.dtm]
        return self._terms_per_doc

    @property
    def reverse_term_dict(self):
        if not hasattr(self, "_reverse_term_dict"):
            self._reverse_term_dict = {v:k for k,v in self.all_terms.items()}
        return self._reverse_term_dict

    def term_existinds(self, use_index=True):
        if not hasattr(self, "_term_existinds"):
            occurs_in = [set(j[0] for j in i) if i else [] for i in self.dtm]
            self._term_existinds = [[ndoc for ndoc, doc in enumerate(occurs_in) if k in doc] for k in self.all_terms.keys()]
        return self._term_existinds if use_index else {self.all_terms[k]: v for k,v in enumerate(self._term_existinds)}

    def as_csr(self):
        if not hasattr(self, "_csr"):
            data = flatten([[elem[1] for elem in row] for row in self.dtm])
            row = flatten([[elem[0] for elem in row] for row in self.dtm])
            col = flatten([[nrow for elem in row] for nrow, row in enumerate(self.dtm)])
            self._csr = csr_matrix((data, (row, col)), shape=(len(self.all_terms), len(self.dtm)))
        return self._csr

    def add_pseudo_keyworddocs(self):
        # see [VISR12: 4.2.1] they create a pseudo-document d_t for each tag
        assert not self.includes_pseudodocs
        self.includes_pseudodocs = True
        max_val = self.as_csr().max()
        self.dtm += [[[i, max_val]] for i in self.all_terms.keys()]
        if hasattr(self, "_csr"): del self._csr
        if hasattr(self, "_term_existinds"): del self._term_existinds


    @staticmethod
    def filter(dtm, min_count, use_n_docs_count=True, verbose=False, descriptions=None, all_terms=None, cap_max=True):
        """can get either a DocTermMatrix as dtm, or a DocTermMatrix.dtm as dtm and an all_terms value"""
        if not all_terms:
            assert isinstance(dtm, DocTermMatrix)
            all_terms = dtm.all_terms
            dtm = dtm.dtm
        else:
            assert not isinstance(dtm, DocTermMatrix)
        if use_n_docs_count:
            occurences = [set(i[0] for i in doc) for doc in dtm]
            term_counts = {term: sum([term in i for i in occurences]) for term in tqdm(all_terms.keys(), desc="Counting Terms")}
        else:
            flat_terms = [flatten([[i[0]] * i[1] for i in doc]) for doc in dtm]
            term_counts = Counter(flatten(flat_terms))
        used_terms = {k: v for k, v in term_counts.items() if v >= min_count}
        if cap_max:
            used_terms = {k: v for k, v in used_terms.items() if v <= len(dtm)-min_count}
        if verbose:
            print(f"Filtering such that terms occur " + (f"in at least {min_count} documents" if use_n_docs_count else
                                                         f"at least {min_count} times") + f", which are {len(used_terms)} of {len(term_counts)} terms.")
            most_used = sorted(list(used_terms.items()), key=lambda x: x[1], reverse=True)[:10]
            print("The most used terms are: " + ", ".join([f"{all_terms[ind]} ({count})" for ind, count in most_used]))
            show_hist(list(used_terms.values()), "Occurences per Keyword", xlabel="Occurences per Keyword", cutoff_percentile=93)
        doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc] for doc in dtm]
        all_terms = {all_terms[elem]: i for i, elem in enumerate(used_terms.keys())}; del used_terms
        doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc if ind in all_terms] for doc in doc_term_matrix]
        assert set(i[0] for doc in doc_term_matrix for i in doc) == set(all_terms.values())
        all_terms = {v: k for k, v in all_terms.items()}
        if descriptions:
            assert all(all_terms[ind] in descriptions._descriptions[ndoc] for ndoc, doc in enumerate(tqdm(doc_term_matrix)) for ind, count in doc)
            if verbose:
                print("Documents without any keyphrase:", [descriptions._descriptions[i] for i, e in enumerate(doc_term_matrix) if len(e) < 1][:5])
                print("Documents with just 1 keyphrase:", [[descriptions._descriptions[i], all_terms[e[0][0]]] for i, e in enumerate(doc_term_matrix) if len(e) == 1][:5])
        return DocTermMatrix(dtm=doc_term_matrix, all_terms=all_terms, quant_name="count")


    def doc_freq(self, keyword, rel=False, supress=False):
        if supress:
            return len(self.term_existinds(use_index=False).get(keyword, [])) / (self.n_docs if rel else 1)
        return len(self.term_existinds(use_index=False)[keyword]) / (self.n_docs if rel else 1)

    def apply_quant(self, quant_name, **kwargs):
        return DocTermMatrix(dtm=apply_quant(quant_name, self, **kwargs), all_terms=self.all_terms, quant_name=quant_name)


    def term_quants(self, term): #returns a list of the quantification (count or whatever it is) for the term
        existinds = self.term_existinds(use_index=False)[term]
        return [0 if i not in existinds else dict(self.dtm[i])[self.reverse_term_dict[term]] for i in range(len(self.dtm))]




def apply_quant(quant, dtm, verbose=False, descriptions=None):
    from derive_conceptualspace.util.text_tools import tf_idf, ppmi
    if quant == "tfidf":
        # quantification = tf_idf(dtm, verbose=verbose, descriptions=descriptions)
        quantification = csr_to_list(TfidfTransformer().fit_transform(dtm.as_csr().T))
    elif quant == "ppmi":
        quantification = ppmi(dtm, verbose=verbose, descriptions=descriptions)
    elif quant == "count":
        quantification = dtm.dtm
    elif quant == "binary":
        quantification = [[[j[0],min(j[1],1)] for j in i] for i in dtm.dtm]
    else:
        raise NotImplementedError()
    assert len(dtm.dtm) == len(quantification)
    assert all(len(i) == len(j) for i, j in zip(dtm.dtm, quantification))
    return quantification



def csr_to_list(csr, vocab=None):
    if not isinstance(csr, csr_matrix):
        csr = csr_matrix(csr)
    aslist = [list(sorted(zip((tmp := csr.getrow(nrow).tocoo()).col, tmp.data), key=lambda x:x[0])) for nrow in range(csr.shape[0])]
    if not vocab:
        return aslist
    return aslist, {v: k for k, v in vocab.items()}



def dtm_dissimmat_loader(quant_dtm, dissim_mat):
    return dtm_loader(quant_dtm), dissim_mat

def dtm_loader(doc_term_matrix):
    dtm = DocTermMatrix.fromstruct(doc_term_matrix[1][1])
    if get_setting("DEBUG"):
        if len(dtm.dtm) > get_setting("DEBUG_N_ITEMS"):
            warnings.warn("len(dtm) > DEBUG_N_ITEMS!!")
    return dtm