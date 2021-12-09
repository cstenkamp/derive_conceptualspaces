from scipy.sparse import csr_matrix
from tqdm import tqdm

flatten = lambda l: [item for sublist in l for item in sublist]


class DocTermMatrix():
    def __init__(self, *args, **kwargs):
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
        else:
            assert False
        assert set(self.all_terms) == set(flatten([[elem[0] for elem in row] for row in self.dtm]))
        print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")

    #num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(dtm.all_terms)))]

    @property
    def doc_freqs(self):
        """the number of documents containing a word, for all words"""
        if not hasattr(self, "_doc_freqs"):
            occurences = [set(i[0] for i in doc) for doc in self.dtm]
            self._doc_freqs = {term: sum(term in doc for doc in occurences) for term in tqdm(list(self.all_terms.keys()))}
            print("Most frequent term:", self.all_terms[max(self._doc_freqs.items(), key=lambda x:x[1])[0]])
        return self._doc_freqs

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