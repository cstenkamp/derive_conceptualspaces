from scipy.sparse import csr_matrix

flatten = lambda l: [item for sublist in l for item in sublist]


class DocTermMatrix():
    def __init__(self, *args):
        self.includes_pseudodocs = False
        if len(args) == 1 and isinstance(args[0], dict):
            assert "doc_term_matrix" in args[0] and "all_terms" in args[0]
            self.dtm = args[0]["doc_term_matrix"]
            self.all_terms = args[0]["all_terms"]
            if isinstance(next(iter(self.all_terms.keys())), str):
                self.all_terms = {int(k): v for k, v in self.all_terms.items()}
            assert set(self.all_terms) == set(flatten([[elem[0] for elem in row] for row in self.dtm]))
            #TODO store meta-info
            #TODO assert dass len(self.dtm) == len(mds_obj.names)
        print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")

    #num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(dtm.all_terms)))]

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