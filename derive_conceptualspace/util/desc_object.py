import random
import textwrap

from .jsonloadstore import Struct
from ..settings import get_setting

flatten = lambda l: [item for sublist in l for item in sublist]

from misc_util.pretty_print import fmt

class Description():
    def __init__(self, text: str, lang: str, title: str, add_title: bool, add_subtitle: bool,
                 orig_textlang: str = None, origlang_text: str = None, subtitle: str = None,
                 orig_titlelang: str = None, origlang_title: str = None, origlang_subtitle: str = None):
        self.text = text
        self.lang = lang
        self.title = title
        self.subtitle = subtitle

        self._add_title = add_title
        self._add_subtitle = add_subtitle
        self._orig_textlang = orig_textlang
        self._origlang_text = origlang_text
        self._orig_titlelang = orig_titlelang
        self._origlang_title = origlang_title
        self._origlang_subtitle = origlang_subtitle

        self.processing_steps = []

    def json_serialize(self):
        return Struct(**self.__dict__)

    @staticmethod
    def fromstruct(struct):
        construct_args = ["text", "lang", "title", "subtitle", "_add_title", "_add_subtitle", "_orig_textlang", "_origlang_text", "_orig_titlelang", "_origlang_title", "_origlang_subtitle"]
        args = {key if not key.startswith("_") else key[1:]: struct[key] for key in construct_args if key in struct}
        desc = Description(**args)
        for key in struct:
            if key not in construct_args:
                setattr(desc, key, struct[key])
        return desc

    @property
    def orig_textlang(self):
        if hasattr(self, "_orig_textlang"):
            return self._orig_textlang
        return self.lang

    @property
    def is_translated(self):
        return self.lang != self.orig_textlang

    @property
    def orig_text(self):
        if not self.is_translated:
            return self.text
        return self._origlang_text

    def __str__(self):
        return (f"Description({self.orig_textlang}: '{((self.title+'. ') if self._add_title else '')}"
                f"{((self.subtitle+'. ') if self._add_subtitle and self.subtitle else '')}{textwrap.shorten(self.text, 70)}')")

    def __repr__(self):
        return fmt(f"Description({self.orig_textlang}: '{(('*b*'+self.title+'*b*. ') if self._add_title else '')}"
                   f"{(('*g*'+self.subtitle+'*g*. ') if self._add_subtitle and self.subtitle else '')}{textwrap.shorten(self.text, 70)}')")


    def process(self, procresult, procname):
        self.processing_steps.append((procresult, procname))

    @property
    def processed_text(self):
        """returns maximally processed text."""
        if len(self.processing_steps) > 0:
            return self.processing_steps[-1][0]
        return self.unprocessed_text

    @property
    def unprocessed_text(self):
        """returns minimally processed text."""
        return ((self.title+". ") if self._add_title else "") + ((self.subtitle+". ") if self._add_subtitle and self.subtitle else "") + self.text

    def processed_as_string(self, no_dots=False):
        sent_join = " " if no_dots else ". "
        if isinstance(self.processed_text, list):
            if isinstance(self.processed_text[0], list):
                return sent_join.join([" ".join(i) for i in self.processed_text])
            else:
                return " ".join(self.processed_text)
        raise NotImplementedError()

    def __contains__(self, item):
        return bool(self.count_phrase(item))

    def count_phrase(self, item):
        if not isinstance(item, str):
            assert False #in the future just return False
        # assert not any(" " in i for i in self.bow.keys()) #TODO add this assertion back once I have a parameter for if I should include n-grams
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


def pp_descriptions_loader(vocab, descriptions):
    descriptions = [Description.fromstruct(i[1][1]) for i in descriptions]
    if get_setting("DEBUG"):
        assert get_setting("RANDOM_SEED", default_none=True)
        random.seed(get_setting("RANDOM_SEED"))
        n_items = get_setting("DEBUG_N_ITEMS")
        assert n_items <= len(descriptions), f"The Descriptions-Dataset contains {len(descriptions)} samples, but you want to draw {n_items}!"
        descriptions = [descriptions[key] for key in random.sample(range(len(descriptions)), k=n_items)]
        vocab = sorted(set(flatten([set(i.bow.keys()) for i in descriptions])))
    return {"vocab": vocab, "descriptions": descriptions}
