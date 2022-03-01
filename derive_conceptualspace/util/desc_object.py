import textwrap
from collections import Counter
from os.path import basename
from typing import Optional, List
import inspect
import logging
from types import FunctionType

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .dtm_object import DocTermMatrix, csr_to_list
from .jsonloadstore import Struct
from .threadworker import WorkerPool
from derive_conceptualspace.settings import get_setting, forbid_setting, get_ncpu

flatten = lambda l: [item for sublist in l for item in sublist]
logger = logging.getLogger(basename(__file__))

from misc_util.pretty_print import fmt

########################################################################################################################
########################################################################################################################
########################################################################################################################

class Description():
    def __init__(self, text, title,
                 lang, orig_textlang=None, origlang_text=None,
                 orig_titlelang=None, origlang_title=None,
                 subtitle=None, origlang_subtitle=None, orig_subtitlelang=None,
                 additionals=None, bow=None):
        self.text = text
        self.lang = lang
        self.title = title
        self.subtitle = subtitle
        self._additionals = additionals
        self._bow = bow

        self._orig_textlang = orig_textlang
        self._origlang_text = origlang_text
        self._orig_titlelang = orig_titlelang
        self._orig_subtitlelang = orig_subtitlelang
        self._origlang_title = origlang_title
        self._origlang_subtitle = origlang_subtitle

        self.processing_steps = []
        self.list_ref: Optional[DescriptionList] = None

    _add_title = property(lambda self: self.list_ref.add_title)
    _add_subtitle = property(lambda self: self.list_ref.add_subtitle)

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
        if hasattr(self, "_orig_textlang") and self._orig_textlang is not None:
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

    def repr(self, shortento=70):
        if self.text is None:
            return f"Description({self.orig_textlang}: {self.title})"
        return f"Description({self.orig_textlang}: '{(('*b*'+self.title+'*b*. ') if self._add_title else '')} {(('*g*'+self.subtitle+'*g*. ') if self._add_subtitle and self.subtitle else '')}{textwrap.shorten(self.text, shortento)}')"

    def __str__(self):
        return self.repr().replace("*b*", "").replace("*g*", "")

    def __repr__(self):
        return fmt(self.repr())


    def process(self, procresult, procname):
        if isinstance(procresult, list):
            procresult = [i for i in procresult if i]
        self.processing_steps.append((procresult, procname))

    @property
    def processed_text(self):
        """returns maximally processed text."""
        if len(self.processing_steps) > 0:
            return self.processing_steps[-1][0]
        return self.unprocessed_text

    @property
    def unprocessed_text(self):
        """returns minimally processed text (translated and with title/subtitle, but no pp steps)"""
        firstpart = ((self.title+". ") if self._add_title else "") + ((self.subtitle+". ") if self._add_subtitle and self.subtitle else "")
        secondpart = self.text if self.text is not None else " ".join([f"{k} "*v for k, v in self.bow().items()])
        return firstpart + secondpart

    def processed_as_string(self, no_dots=False, allow_shorten=False, insert_sentborder=False):
        #sentborder allows to recognize keywords that would be detected across sentences
        if allow_shorten and len(self.processing_steps) == 0 and not self._add_subtitle and not self._add_subtitle:
            return " ".join([k for k in self.bow().keys()])
        sent_join = " xxMA_SENTBORDERxx " if insert_sentborder else " " if no_dots else ". "
        if isinstance(self.processed_text, list):
            if isinstance(self.processed_text[0], list):
                return sent_join.join([" ".join(i) for i in self.processed_text])
            else:
                return " ".join(self.processed_text)
        raise NotImplementedError()

    def __contains__(self, item):
        return bool(self.count_phrase(item))

    def count_phrase(self, item):
        """this function is used in kw-extraction (which makes a doc-term-matrix from it) and dissim-matrix"""
        assert isinstance(item, str)
        #pp-descriptions NEVER CREATES NGRAMS, also not in it's BoW.
        # ...the only exception (and reason for the stuff after the `or`) is that for if sklearn's countvectorizer does something different then our word-tokenization (in postprocess-candidates)
        if (not " " in item) or (" " in item and item in self._bow): #latter one is only bc of the problem when lemmatizing and letting sklearn's count overwrite it so we're force-writing the sklearn-count into the bow for asstions in postprocess_cands
            return self.bow().get(item, 0) #the bow contains ALL words of the desription (so min_df==1)
        return occurrences(" "+self.processed_as_string()+" ", " "+item+" ") + (occurrences(" "+self.processed_as_string()+" ", " "+item+". ") if "sent_tokenize" in self.list_ref.proc_steps else 0)
        #reason for the second summand: Keyphrases like 'abschlussprufung ende 2' ARE inside a (badly sent-tokenized) string that goes "... ende 2. Semester"
        # TODO check the following things:
        #   * text contains: "deutschen sprachen deutschen sprachen deutschen sprache", candidate is "deutschen sprache" -> correct is to return 1, however I DISLIKE this.


    def bow(self):
        if not hasattr(self, "_bow") or self._bow is None:
            if isinstance(self.processed_text[0], list):
                self._bow = Counter(flatten(self.processed_text))
            else:
                self._bow = Counter(self.processed_text)
        return self._bow

    def n_words(self):
        if hasattr(self, "_bow"):
            return sum(self.bow().values())
        if isinstance(self.processed_text, str):
            return self.processed_text.count(" ")+1
        elif isinstance(self.processed_text[0], str):
            return len(self.processed_text)
        elif isinstance(self.processed_text[0][0], str):
            return len(flatten(self.processed_text))


def occurrences(string, sub):
    # https://stackoverflow.com/a/2970542/5122790
    # reason for this: `" fruhen neuzeit fruhen neuzeit ".count(" "+"fruhen neuzeit"+" ") == 1` which is unintuitive as fuck
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count


########################################################################################################################
########################################################################################################################
########################################################################################################################

class DescriptionList():
    def __init__(self, add_title, add_subtitle, translate_policy, additionals_names):
        self.add_title = add_title
        self.add_subtitle = add_subtitle
        self.translate_policy = translate_policy
        self.additionals_names = additionals_names
        self.proc_steps = []
        self._descriptions: List[Description] = []

    def json_serialize(self):
        return Struct(**{k: v if not isinstance(v, set) else tuple(v) for k, v in self.__dict__.items()})

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
        return obj

    def add(self, desc):
        desc.list_ref = self
        self._descriptions.append(desc)
        if hasattr(self, "_all_words"): del self._all_words

    def __len__(self):
        return len(self._descriptions)

    def confirm(self, confirmwhat, **kwargs):
        if confirmwhat == "translate_policy":
            assert "language" in kwargs
            assert all((i.is_translated and i.lang == kwargs["language"] and i.orig_textlang != kwargs["language"]) or
                       (not i.is_translated and i.orig_textlang == i.lang and i.orig_textlang == kwargs["language"]) for i in self._descriptions)
        else:
            assert False, f"cannot confirm {confirmwhat}"

    def process_all(self, proc_fn, proc_name, proc_base=None, indiv_kwargs=None, pgbar="", multiproc=False):
        if not multiproc:
            for desc in tqdm(self._descriptions, desc=pgbar if isinstance(pgbar, str) else None) if pgbar else self._descriptions:
                kwargs = {k: v(desc) for k, v in indiv_kwargs.items()} if indiv_kwargs is not None else {}
                base = desc.processed_text if proc_base is None else proc_base(desc)
                desc.process(proc_fn(base, **kwargs), proc_name)
        else:
            # Times: lemmatization, german, i7octa, 445 elems, X procs: 1: 1:41, 1:38, 3: 0:48, 6: 0:36, 0:32, 12: 0:31, 0:35, 18: 0:30
            if isinstance(proc_fn, FunctionType): raise NotImplementedError("So far only implemented for the case where proc_fn is a class with __call__")
            if indiv_kwargs is not None: raise NotImplementedError()
            bases = [desc.processed_text if proc_base is None else proc_base(desc) for desc in self._descriptions]
            with WorkerPool(get_ncpu(ignore_debug=True), proc_fn, pgbar=pgbar) as pool:
                res, interrupted = pool.work(bases, lambda proc_fn, elem: proc_fn(elem))
            assert not interrupted
            for n, desc in enumerate(self._descriptions):
                desc.process(res[n], proc_name)
        self.proc_steps.append(proc_name)

    def unprocessed_texts(self, remove_htmltags=False):
        for desc in self._descriptions:
            if remove_htmltags and desc.processing_steps[0][1] == "remove_htmltags":
                yield desc.processing_steps[0][0]
            else:
                yield desc.unprocessed_text

    def all_words(self):
        if not hasattr(self, "_all_words"):
            if isinstance(self._descriptions[0].processed_text[0], str):
                self._all_words = set(flatten([i.processed_text for i in self._descriptions]))
            else:
                self._all_words = set(flatten([flatten(i.processed_text) for i in self._descriptions]))
        return self._all_words


    def generate_DocTermMatrix(self, min_df=1, max_ngram=None, do_tfidf=None):
        if self.proc_steps[-1] == "bow":
            assert max_ngram in [None, 1], "Can't do!"
            print("Preprocessed produced a bag-of-words already. Config `max_ngram` is useless!")
            forbid_setting("max_ngram")
            all_words = dict(enumerate(set(flatten(i.bow().keys() for i in self._descriptions))))
            rev = {v: k for k, v in all_words.items()}
            dtm = [[[rev[k], v] for k, v in i.bow().items()] for i in self._descriptions]
            dtm = DocTermMatrix(dtm=dtm, all_terms=all_words, quant_name="count")
            if min_df > 1:
                dtm = DocTermMatrix.filter(dtm, min_df, use_n_docs_count=get_setting("CANDS_USE_NDOCS_COUNT"), verbose=get_setting("VERBOSE"), descriptions=self)
            return dtm, {"ngrams_in_embedding": False}
        elif hasattr(self, "recover_settings"):
            from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents, get_countvec
            pp_comps = PPComponents.from_str(self.recover_settings["pp_components"])
            if pp_comps.use_skcountvec:
                cnt = get_countvec(**self.recover_settings, max_ngram=(max_ngram or 1), min_df=min_df)
                fit_base = lambda: self.unprocessed_texts(remove_htmltags=pp_comps.remove_htmltags)
            else: raise NotImplementedError()
        else:
            cnt = CountVectorizer(strip_accents=None, lowercase=False, stop_words=None, ngram_range=(1, (max_ngram or 1)), min_df=min_df, token_pattern=r"(?u)(\b\w\w+\b|\b\d\b)") #this token_pattern allows for single-digit-numbers
            fit_base = lambda: self.iter("processed_as_string", insert_sentborder=("sent_tokenize" in self.proc_steps))
            #if we sent_tokenize, we have something like "2. Semester" in the original, which becomes due to nltk's suckiness [["bla", "2"], ["Semester", "blub"]]. It shouldn't find stuff across sentence borders, so we insert detectables strings that we can remove later.
            # TODO If I can do sent_tokenize for the CountVectorizer I need to update this here as well!
        if do_tfidf is not None:
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
            pipe = Pipeline([("count", cnt), ("tfidf", TfidfTransformer(use_idf=(do_tfidf=="tfidf")))]).fit(fit_base())
            aslist, all_words = csr_to_list(pipe.transform(fit_base()), pipe["count"].vocabulary_)
            return DocTermMatrix(dtm=aslist, all_terms=all_words, quant_name=do_tfidf), {"ngrams_in_embedding": any(" " in i for i in all_words.values()), "sklearn_tfidf": True}
        X = cnt.fit_transform(fit_base())
        aslist, all_words = csr_to_list(X, cnt.vocabulary_)
        return DocTermMatrix(dtm=aslist, all_terms=all_words, quant_name="count"), {"ngrams_in_embedding": any(" " in i for i in all_words.values())}


    def add_embeddings(self, embeddings):
        for desc, embedding in zip(self._descriptions, list(embeddings)):
            desc.embedding = embedding

    def iter(self, func, **kwargs):
        for desc in self._descriptions:
            yield getattr(desc, func)(**kwargs)

    def filter_words(self, min_words):
        tmp = [i for i in self._descriptions if i.n_words() >= min_words]
        print(f"Removed {len(self)-len(tmp)} of {len(self)} Descriptions because they are less than {min_words} words ({len(tmp)} left)")
        self._descriptions = tmp
        return self

    def __getitem__(self, item):
        return next(i for i in self._descriptions if i.title == item)