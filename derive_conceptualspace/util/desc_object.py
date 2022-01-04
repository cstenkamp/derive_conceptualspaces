import random
import textwrap
from collections import Counter
from typing import Optional, List
import inspect

from tqdm import tqdm

from .dtm_object import DocTermMatrix
from .jsonloadstore import Struct
from ..settings import get_setting


flatten = lambda l: [item for sublist in l for item in sublist]

from misc_util.pretty_print import fmt

########################################################################################################################
########################################################################################################################
########################################################################################################################

class Description():
    def __init__(self, text, lang, title,
                 orig_textlang=None, origlang_text=None, subtitle=None, orig_titlelang=None, origlang_title=None, origlang_subtitle=None):
        self.text = text
        self.lang = lang
        self.title = title
        self.subtitle = subtitle

        self._orig_textlang = orig_textlang
        self._origlang_text = origlang_text
        self._orig_titlelang = orig_titlelang
        self._origlang_title = origlang_title
        self._origlang_subtitle = origlang_subtitle

        self.processing_steps = []
        self.list_ref: Optional[DescriptionList] = None

    _add_title = property(lambda self: self.list_ref.add_title)
    _add_subtitle = property(lambda self: self.list_ref.add_subtitle)
    _includes_ngrams = property(lambda self: self.list_ref.proc_has_ngrams)
    _proc_min_df = property(lambda self: self.list_ref.proc_min_df)
    _proc_ngram_range = property(lambda self: self.list_ref.proc_ngram_range) # ngram_range > 1 means the bow is incomplete!

    def json_serialize(self):
        return Struct(**{k: v for k,v in self.__dict__.items() if k != "list_ref"})

    @staticmethod
    def fromstruct(struct):
        construct_args = ["text", "lang", "title", "subtitle", "_orig_textlang", "_origlang_text", "_orig_titlelang", "_origlang_title", "_origlang_subtitle"]
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
        #So, we need this for two things: kw-extraction and dissimilarity-matrix. In the former, we'll only take ones that occur often enough anyway,
        #           and in the latter...?
        #Problem: Es kann sein, dass processed_phrase filtered ist, sodass ..? #TODO argh
        # Das Problem ist ja, ich erzeuge erst die n-grams und lösche sie danach raus wenn sie nicht oft genug in anderen descriptions vorkommt.
        # Das heißt, ich kann einen Teil der very description um die es geht nehmen und wenn sie nicht oft genug insgesamt vorkommt würde count_phrase = 0 sein
        if not isinstance(item, str):
            assert False #in the future just return False
        if self._proc_min_df == 1:
            if (self._includes_ngrams and " " in item) or not " " in item:
                return self.bow().get(item, 0)
        if item in self.bow():
            return self.bow()[item]
        if self._proc_min_df == 1:
            if " " in item:
                return self.processed_as_string().count(item)
            return self.processed_text.count(item)
        raise NotImplementedError()
        #TODO I don't feel like this is done...! I should be able to check if it's in there even if the doc-term-matrix forgot that its in there


        if not " " in item:
            return self.bow().get(item, 0)
        elif self._includes_ngrams and item in self.bow():
            return self.bow()[item]
        else:
            raise NotImplementedError()
        # assert not any(" " in i for i in self.bow.keys()) #TODO add this assertion back once I have a parameter for if I should include n-grams
        # if " " in item: #TODO why did I even need this originally? When did I check for items with spaces?
        #     items = item.split(" ")
        #     for it in items:
        #         if it not in self.bow:
        #             return 0
        #     if item in self.processed_as_string():
        #         return self.processed_as_string().count(item)
        #     elif item in self.processed_as_string(no_dots=True):
        #         return 0 # this is legit not a candidate, but I want to be able to set breakpoints in cases where this is not the reason
        return self.bow.get(item, 0)

    def bow(self):
        if not hasattr(self, "_bow"):
            if isinstance(self.processed_text[0], list):
                self._bow = Counter(flatten(self.processed_text))
            else:
                self._bow = Counter(self.processed_text)
        return self._bow

########################################################################################################################
########################################################################################################################
########################################################################################################################

class DescriptionList():
    def __init__(self, add_title, add_subtitle, translate_policy):
        self.add_title = add_title
        self.add_subtitle = add_subtitle
        self.translate_policy = translate_policy
        self.proc_has_ngrams = False
        self.proc_steps = []
        self._descriptions: List[Description] = []

    def json_serialize(self):
        return Struct(**self.__dict__)

    @staticmethod
    def from_json(descriptions):
        #TODO the pp_descriptions_loader (until commit 5c90a2043eb) allowed to get a subset of the descriptions if DEBUG
        assert descriptions[0] == "DescriptionList" and descriptions[1][0] == "Struct"
        init_args = inspect.getfullargspec(DescriptionList.__init__).args[1:]
        assert descriptions[1][1].keys() >= set(init_args)
        obj = DescriptionList(**{k: v for k,v in descriptions[1][1].items() if k in init_args})
        for k, v in {k: v for k, v in descriptions[1][1].items() if k not in init_args + ["_descriptions"]}.items():
            setattr(obj, k, v)
        for desc in descriptions[1][1]["_descriptions"]:
            assert desc[0] == "Description" and desc[1][0] == "Struct"
            obj.add(Description.fromstruct(desc[1][1]))
        if get_setting("NGRAMS_IN_EMBEDDING", silent=True) and get_setting("MAX_NGRAM") > 1:
            assert obj.proc_has_ngrams
        return obj

    def add(self, desc):
        desc.list_ref = self
        self._descriptions.append(desc)
        if hasattr(self, "_all_words"): del self._all_words

    def __len__(self):
        return len(self._descriptions)

    def confirm(self, confirmwhat):
        if confirmwhat == "translate_policy":
            assert all((i.is_translated and i.lang == "en" and i.orig_textlang != "en") or
                       (not i.is_translated and i.orig_textlang == i.lang and i.orig_textlang == "en") for i in self._descriptions)
        else:
            assert False, f"cannot confirm {confirmwhat}"

    def process_all(self, proc_fn, proc_name, adds_ngrams=False, proc_base=None, indiv_kwargs=None, pgbar=""):
        for desc in tqdm(self._descriptions, desc=pgbar if isinstance(pgbar, str) else None) if pgbar else self._descriptions:
            kwargs = {k: v(desc) for k, v in indiv_kwargs.items()} if indiv_kwargs is not None else {}
            base = desc.processed_text if proc_base is None else proc_base(desc)
            desc.process(proc_fn(base, **kwargs), proc_name)
        self.proc_steps.append(proc_name)
        self.proc_has_ngrams = self.proc_has_ngrams or adds_ngrams

    @property
    def processed_texts(self):
        for desc in self._descriptions:
            yield desc.processed_text

    @property
    def unprocessed_texts(self):
        for desc in self._descriptions:
            yield desc.unprocessed_text

    def all_words(self):
        assert get_setting("NGRAMS_IN_EMBEDDING", silent=True) == self.proc_has_ngrams
        if not hasattr(self, "_all_words"):
            if isinstance(self._descriptions[0].processed_text[0], str):
                self._all_words = set(flatten([i.processed_text for i in self._descriptions]))
            else:
                self._all_words = set(flatten([flatten(i.processed_text) for i in self._descriptions]))
        return self._all_words


    def generate_DocTermMatrix(self, min_df=1, max_ngram=None):
        if hasattr(self, "recover_settings"):
            from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents, get_countvec
            pp_components = PPComponents.from_str(self.recover_settings["pp_components"])
            if pp_components.use_skcountvec:
                max_ngram = max_ngram or int(self.recover_settings["max_ngram"])
                cnt = get_countvec(pp_components, max_ngram, min_df)
                X = cnt.fit_transform(self.unprocessed_texts)
                aslist = [list(sorted(zip((tmp := X.getrow(nrow).tocoo()).col, tmp.data), key=lambda x:x[0])) for nrow in range(X.shape[0])]
                all_words = {v: k for k, v in cnt.vocabulary_.items()}
                return DocTermMatrix(dtm=aslist, all_terms=all_words)
        else:
            return DocTermMatrix.from_descriptions(self, min_df=min_df)
