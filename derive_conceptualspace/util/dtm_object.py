import warnings
from collections import Counter
from functools import lru_cache

from plotly.serializers import np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from scipy.spatial.distance import squareform

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
        assert descriptions.proc_min_df <= min_df #TODO!
        all_terms = {n: elem for n, elem in enumerate(vocab)}
        reverse_term_dict = {v: k for k,v in all_terms.items()}
        dtm = []
        for desc in tqdm(descriptions._descriptions, desc="Loading Bag-Of-Words to DocTermMatrix.."):
            dtm.append([[reverse_term_dict[k], v] for k,v in desc.bow().items()])
        if min_df > 1:
            return DocTermMatrix.filter(dtm, min_df, use_n_docs_count=True, verbose=verbose, descriptions=descriptions, all_terms=all_terms) #TODO make use_n_docs_count an arg
        return DocTermMatrix(dtm, all_terms, quant_name="count")

    def __init__(self, dtm, all_terms, quant_name, verbose=False):
        self.includes_pseudodocs = False
        self.dtm = dtm
        self.all_terms = {n: elem for n, elem in enumerate(all_terms)} if isinstance(all_terms, list) else all_terms
        self.quant_name = quant_name
        # if "all_terms" in kwargs and "descriptions" in kwargs: assert hasattr(kwargs["descriptions"][0], "bow")
        # for desc in kwargs["descriptions"]: self.dtm.append([[self.reverse_term_dict[k], v] for k,v in desc.bow().items()])
        assert set(self.all_terms) == set(flatten([[elem[0] for elem in row] for row in self.dtm]))
        if verbose:
            print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")
            self.show_info()

    def json_serialize(self):
        return Struct(**{k:v for k,v in self.__dict__.items() if not k.startswith("_") and k not in ["csr_matrix", "doc_freqs", "reverse_term_dict"]})

    n_docs = property(lambda self: len(self.dtm))

    def show_info(self, descriptions=None):
        occurs_in = [set(j[0] for j in i) if i else set() for i in self.dtm]
        num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(self.all_terms)), desc="Counting Occurences [verbose]")]
        show_hist(num_occurences, f"Docs per Keyword ({self.n_docs} docs, {len(self.all_terms)} terms)", xlabel="# Documents the Keyword appears in", ylabel="Count (log scale)", cutoff_percentile=98, log=True)
        above_threshold = len([i for i in num_occurences if i>= get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)])
        sorted_canditerms = sorted([[ind, elem] for ind, elem in enumerate(num_occurences)], key=lambda x:x[1], reverse=True)
        print(f"Found {len(self.all_terms)} candidate Terms, {above_threshold} ({round(above_threshold/len(self.all_terms)*100)}%) of which occur in at least {get_setting('CANDIDATE_MIN_TERM_COUNT', silent=True)} descriptions.")
        print("The 25 terms that occur in the most descriptions (incl the #descriptions they occur in):",
              ", ".join([f"{self.all_terms[ind]} ({occs})" for ind, occs in sorted_canditerms[:25]]))
        if descriptions is not None:
            max_ind = np.unravel_index(self.as_csr().argmax(), self.as_csr().shape)
            print(f"Max value: Term *b*{self.all_terms[max_ind[0]]}*b* has value *b*{dict(self.dtm[max_ind[1]])[max_ind[0]]:.3f}*b* for doc *b*{descriptions._descriptions[max_ind[1]].title}*b*")

    def doc_freqs(self, verbose=False):
        """the number of documents containing a word, for all words"""
        if not hasattr(self, "_doc_freqs"):
            # occurences = [set(i[0] for i in doc) for doc in self.dtm]
            # self._doc_freqs = {term: sum(term in doc for doc in occurences) for term in tqdm(list(self.all_terms.keys()), desc="Calculating Doc-Frequencies")}
            self._doc_freqs = dict(enumerate(self.as_csr(binary=True).sum(axis=1).squeeze().tolist()[0]))
            if verbose:
                most_freq = sorted(self._doc_freqs.items(), key=lambda x:x[1], reverse=True)[:5]
                print("Most frequent terms:", ", ".join([f"{self.all_terms[term]} ({num})" for term, num in most_freq]))
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
            self._term_existinds = {k: [ndoc for ndoc, doc in enumerate(occurs_in) if k in doc] for k in self.all_terms.keys()}
        return self._term_existinds if use_index else {self.all_terms[k]: v for k,v in self._term_existinds.items()}

    @lru_cache
    def as_csr(self, binary=False):
        if binary:
            data = flatten([[1 for _ in row] for row in self.dtm])
        else:
            data = flatten([[elem[1] for elem in row] for row in self.dtm])
        row = flatten([[elem[0] for elem in row] for row in self.dtm])
        col = flatten([[nrow for _ in row] for nrow, row in enumerate(self.dtm)])
        return csr_matrix((data, (row, col)), shape=(len(self.all_terms), len(self.dtm)))

    def add_pseudo_keyworddocs(self):
        # see [VISR12: 4.2.1] they create a pseudo-document d_t for each tag
        assert not self.includes_pseudodocs
        self.includes_pseudodocs = True
        max_val = self.as_csr().max()
        self.dtm += [[[i, max_val]] for i in self.all_terms.keys()]
        if hasattr(self, "_csr"): del self._csr
        if hasattr(self, "_term_existinds"): del self._term_existinds


    @staticmethod
    def filter(dtm, min_count, use_n_docs_count=True, verbose=False, descriptions=None, cap_max=True):
        """accepts only a DocTermMatrix as input from now on. descriptions only for verbosity"""
        assert isinstance(dtm, DocTermMatrix)
        if use_n_docs_count:
            term_counts = dtm.doc_freqs()
        else:
            flat_terms = [flatten([[i[0]] * i[1] for i in doc]) for doc in dtm]
            term_counts = Counter(flatten(flat_terms))
        used_terms = {k: v for k, v in term_counts.items() if v >= min_count}
        if cap_max:
            used_terms = {k: v for k, v in used_terms.items() if v <= dtm.n_docs-min_count}
        if verbose:
            print(f"Filtered such that terms occur " + (f"in at least {min_count} documents" if use_n_docs_count else
                                                        f"at least {min_count} times") + f", which are {len(used_terms)} of {len(term_counts)} terms.")
            most_used = sorted(list(used_terms.items()), key=lambda x: x[1], reverse=True)[:10]
            print("The most used terms are: " + ", ".join([f"{dtm.all_terms[ind]} ({count})" for ind, count in most_used]))
            # show_hist(list(used_terms.values()), f"{'Docs' if use_n_docs_count else 'Occurences'} per Keyword ({dtm.n_docs} docs, {len(used_terms)} terms)", xlabel="Occurences per Keyword", cutoff_percentile=93)
            # showing hist here is the same as just having verbose=True in the DTM-Constructor in the last line of this func!
        used_terms_set = set(used_terms.keys())
        all_terms_new = dict(enumerate([v for k, v in dtm.all_terms.items() if k in used_terms_set]))
        all_terms_new_rev = {v: k for k, v in all_terms_new.items()}
        dtm_translator = {k: all_terms_new_rev[v] for k, v in dtm.all_terms.items() if k in used_terms_set}
        doc_term_matrix = [[[dtm_translator.get(ind), num] for ind, num in doc if ind in used_terms_set] for doc in dtm.dtm]
        if descriptions:
            if get_setting("DO_SANITYCHECKS"):
                expected_bows = {ndoc: {all_terms_new[elem]: count for elem, count in doc} for ndoc, doc in enumerate(doc_term_matrix[:10])}
                assert all(all(v==descriptions._descriptions[i].bow()[k] for k, v in expected_bows[i].items() if not " " in k) for i in range(10))
                assert all(all(v==descriptions._descriptions[i].count_phrase(k) for k, v in expected_bows[i].items() if not " " in k) for i in range(10))
                assert all(all_terms_new[ind] in descriptions._descriptions[ndoc] for ndoc, doc in enumerate(tqdm(doc_term_matrix, desc="Cross-checking filtered DCM with Descriptions [sanity-check]")) for ind, count in doc)
            if verbose:
                shown = []
                for n_keyphrases in [0, 1, 20]:
                    items = [[descriptions._descriptions[i], [all_terms_new[j[0]] for j in e]] for i, e in enumerate(doc_term_matrix) if len(e) <= n_keyphrases]
                    if items:
                        print(f"Documents with max {n_keyphrases} keyphrases ({len(items)}):\n  "+"\n  ".join(f"{i[0]}: {', '.join(i[1])}" for i in [j for j in items if j[0] not in shown][:5][:5]))
                        shown += [i[0] for i in items]
        return DocTermMatrix(dtm=doc_term_matrix, all_terms=all_terms_new, quant_name="count", verbose=verbose)

    def apply_quant(self, quant_name, **kwargs):
        dtm = DocTermMatrix(dtm=apply_quant(quant_name, self, **kwargs), all_terms=self.all_terms, quant_name=quant_name)
        if kwargs.get("verbose"):
            dtm.show_info(descriptions=kwargs.get("descriptions"))
        return dtm

    def term_quants(self, term): #returns a list of the quantification (count or whatever it is) for the term
        """Note that is only useful if you do it for 1-3 terms. If you want to do it for ALL, note that
           `dcm.term_quants(terms[0]) == list(dcm.as_csr()[0,:].toarray().squeeze())`"""
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
    quant_dm = dtm_loader(quant_dtm)
    if dissim_mat.ndim == 1:
        dissim_mat = squareform(dissim_mat)
    return quant_dm, dissim_mat

def dtm_loader(doc_term_matrix):
    dtm = DocTermMatrix.fromstruct(doc_term_matrix[1][1])
    if get_setting("DEBUG"):
        if len(dtm.dtm) > get_setting("DEBUG_N_ITEMS"):
            warnings.warn("len(dtm) > DEBUG_N_ITEMS!!")
    return dtm