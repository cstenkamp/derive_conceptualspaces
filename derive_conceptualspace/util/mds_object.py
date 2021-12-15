from copy import deepcopy
from dataclasses import dataclass
from typing import List
import textwrap

from .jsonloadstore import Struct

class Description():
    #TODO add working json-serialize-way

    def __init__(self, text: str, lang: str, for_name: str = None, orig_lang: str = None, orig_text: str = None):
        self.text = text
        self.lang = lang
        self.for_name = for_name
        self._orig_lang = orig_lang
        self._orig_text = orig_text
        self.processing_steps = []

    @staticmethod
    def fromstruct(struct):
        construct_args = ["text", "lang", "for_name", "orig_lang", "orig_text", "_orig_lang", "_orig_text"]
        args = {key if not key.startswith("_") else key[1:]: struct[key] for key in construct_args if key in struct}
        desc = Description(**args)
        for key in struct:
            if key not in construct_args:
                setattr(desc, key, struct[key])
        return desc

    @property
    def orig_lang(self):
        if hasattr(self, "_orig_lang"):
            return self._orig_lang
        return self.lang

    @property
    def is_translated(self):
        return self.lang != self.orig_lang

    @property
    def orig_text(self):
        if not self.is_translated:
            return self.text
        return self._orig_text

    def __repr__(self):
        return f"Description({self.orig_lang}: '{textwrap.shorten(self.text, 70)}')"


    def process(self, procresult, procname):
        self.processing_steps.append((procresult, procname))

    @property
    def processed_text(self):
        """returns maximally processed text."""
        return self.processing_steps[-1][0] if len(self.processing_steps) > 0 else self.text

    def processed_as_string(self, no_dots=False):
        sent_join = " " if no_dots else ". "
        if isinstance(self.processed_text, list):
            if isinstance(self.processed_text[0], list):
                return sent_join.join([" ".join(i) for i in self.processed_text])
        raise NotImplementedError()

    def __contains__(self, item):
        return bool(self.count_phrase(item))

    def count_phrase(self, item):
        if not isinstance(item, str):
            assert False #in the future just return False
        assert not any(" " in i for i in self.bow.keys())
        if " " in item:
            items = item.split(" ")
            for it in items:
                if it not in self.bow:
                    return 0
            if item in self.processed_as_string():
                return self.processed_as_string().count(item)
            elif item in self.processed_as_string(no_dots=True):
                return 0 # this is legit not a candidate, but I want to be able to set breakpoints in cases where this is not the reason
        else:
            return self.bow.get(item, 0)
        return 0 #TODO set breakpoint here for candidate postprocessing! THEN it should NEVER get here!!!


# @dataclass
# class MDSObject:
#     names: List[str]
#     descriptions: List[str] #TODO type Description!!
#     mds: Struct
#     languages: List[str]
#     translate_policy: int
#     orig_n_samples: int
#     original_descriptions: List[str]
#
#     def __post_init__(self):
#         assert len(self.names) == len(self.descriptions)
#         assert self.mds.dissimilarity_matrix_.shape[0] == len(self.names)
#         if self.mds.dissimilarity_matrix_.shape[0] != self.mds.dissimilarity_matrix_.shape[1]:
#             print("The dissimiliarity-matrix was trained on a larger corpus than is used now!")
#             if self.translate_policy == TRANSL:
#                 print("You seem to be less than all descriptions because of the translate-policy, but as the length of the MDS corpus is longer, "
#                       "it seems that the corpus was trained with another translate-policy!")
#
#     def description_of(self, what):
#         if isinstance(what, int):
#             return self.descriptions[what]
#         return self.descriptions[self.index_of(what)]
#
#     def name_of(self, index):
#         return self.names[index]
#
#     def index_of(self, name):
#         return self.names.index(name)
#
#     def __repr__(self):
#         return f"MDSObject({len(self.names)} entries)"