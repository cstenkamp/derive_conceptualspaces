#doc-term-matrix object


class DocTermMatrix():
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            assert "doc_term_matrix" in args[0] and "all_terms" in args[0]
            self.dtm = args[0]["doc_term_matrix"]
            self.all_terms = args[0]["all_terms"]
            if isinstance(next(iter(self.all_terms.keys())), str):
                self.all_terms = {int(k): v for k, v in self.all_terms.items()}
            #TODO store meta-info
        print(f"Loaded Doc-Term-Matrix with {len(self.dtm)} documents and {len(self.all_terms)} items.")

    #num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(dtm.all_terms)))]

    def term_existinds(self, use_index=True):
        if not hasattr(self, "_term_existinds"):
            occurs_in = [set(j[0] for j in i) if i else [] for i in self.dtm]
            self._term_existinds = [[ndoc for ndoc, doc in enumerate(occurs_in) if k in doc] for k in self.all_terms.keys()]
        return self._term_existinds if use_index else {self.all_terms[k]: v for k,v in enumerate(self._term_existinds)}
