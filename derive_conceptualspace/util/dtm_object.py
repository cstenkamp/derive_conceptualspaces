from scipy.sparse import csr_matrix
from tqdm import tqdm

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.jsonloadstore import Struct
from derive_conceptualspace.util.mpl_tools import show_hist

flatten = lambda l: [item for sublist in l for item in sublist]


class DocTermMatrix():

    @staticmethod
    def fromstruct(struct):
        assert not struct["includes_pseudodocs"], "TODO"
        return DocTermMatrix({"doc_term_matrix": struct["dtm"], "all_terms": struct["all_terms"]})


    def __init__(self, *args, verbose=False, **kwargs):
        self.includes_pseudodocs = False
        if len(args) == 1 and isinstance(args[0], dict):
            assert "doc_term_matrix" in args[0] and "all_terms" in args[0]
            assert not kwargs
            self.dtm = args[0]["doc_term_matrix"]
            self.all_terms = args[0]["all_terms"]
            if isinstance(next(iter(self.all_terms.keys())), str):
                self.all_terms = {int(k): v for k, v in self.all_terms.items()}
            #TODO store meta-info
            #TODO assert dass len(self.dtm) == len(mds_obj.names)
        elif "all_terms" in kwargs and "descriptions" in kwargs:
            assert hasattr(kwargs["descriptions"][0], "bow")
            if isinstance(kwargs["all_terms"], dict):
                self.all_terms = kwargs["all_terms"]
            else:
                self.all_terms = {n: elem for n, elem in enumerate(kwargs["all_terms"])}
            self.dtm = []
            for desc in kwargs["descriptions"]:
                self.dtm.append([[self.reverse_term_dict[k], v] for k,v in desc.bow.items()])
        elif "all_phrases" in kwargs and "descriptions" in kwargs and "dtm" in kwargs:
            self.all_terms = {n: elem for n, elem in enumerate(kwargs["all_phrases"])}
            self.dtm = kwargs["dtm"]
            self.descriptions = kwargs["descriptions"]
        else:
            assert False
        assert set(self.all_terms) == set(flatten([[elem[0] for elem in row] for row in self.dtm]))
        print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")
        if verbose:
            self.show_info()

    def json_serialize(self):
        return Struct(**{k:v for k,v in self.__dict__.items() if not k.startswith("_") and k not in ["csr_matrix", "doc_freqs", "reverse_term_dict"]})

    def show_info(self):
        occurs_in = [set(j[0] for j in i) if i else [] for i in self.dtm]
        num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(self.all_terms)))]
        show_hist(num_occurences, "Docs per Keyword", xlabel="# Documents the Keyword appears in", ylabel="Count (log scale)", cutoff_percentile=98, log=True)
        above_threshold = len([i for i in num_occurences if i>= get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)])
        sorted_canditerms = sorted([[ind, elem] for ind, elem in enumerate(num_occurences)], key=lambda x:x[1], reverse=True)
        print(f"Found {len(self.all_terms)} candidate Terms, {above_threshold} ({round(above_threshold/len(self.all_terms)*100)}%) of which occur in at least {get_setting('CANDIDATE_MIN_TERM_COUNT', silent=True)} descriptions.")
        print("The 25 terms that occur in the most descriptions (incl the #descriptions they occur in):",
              ", ".join([f"{self.all_terms[ind]} ({occs})" for ind, occs in sorted_canditerms[:25]]))

    #num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(dtm.all_terms)))]

    @property
    def doc_freqs(self):
        """the number of documents containing a word, for all words"""
        if not hasattr(self, "_doc_freqs"):
            occurences = [set(i[0] for i in doc) for doc in self.dtm]
            self._doc_freqs = {term: sum(term in doc for doc in occurences) for term in tqdm(list(self.all_terms.keys()))}
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


def dtm_dissimmat_loader(quant_dtm, dissim_mat):
    return dtm_loader(quant_dtm), dissim_mat

def dtm_loader(doc_term_matrix):
    dtm = DocTermMatrix.fromstruct(doc_term_matrix[1][1])
    if get_setting("DEBUG"):
        assert len(dtm.dtm) == get_setting("DEBUG_N_ITEMS") #TODO if not
    return dtm