from dataclasses import dataclass
from typing import List
from src.main.load_data.siddata_data_prep.jsonloadstore import Struct

ORIGLAN = 1
ONLYENG = 2
TRANSL = 3

@dataclass
class MDSObject:
    names: List[str]
    descriptions: List[str]
    mds: Struct
    languages: List[str]
    translate_policy: int
    orig_n_samples: int
    original_descriptions: List[str]

    def __post_init__(self):
        assert len(self.names) == len(self.descriptions)
        assert self.mds.dissimilarity_matrix_.shape[0] == len(self.names)
        if self.mds.dissimilarity_matrix_.shape[0] != self.mds.dissimilarity_matrix_.shape[1]:
            print("The dissimiliarity-matrix was trained on a larger corpus than is used now!")
            if self.translate_policy == TRANSL:
                print("You seem to be less than all descriptions because of the translate-policy, but as the length of the MDS corpus is longer, "
                      "it seems that the corpus was trained with another translate-policy!")

    def description_of(self, what):
        if isinstance(what, int):
            return self.descriptions[what]
        return self.descriptions[self.index_of(what)]

    def name_of(self, index):
        return self.names[index]

    def index_of(self, name):
        return self.names.index(name)

    def __repr__(self):
        return f"MDSObject({len(self.names)} entries)"