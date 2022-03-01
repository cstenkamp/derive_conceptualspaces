import warnings
from os.path import basename
import re
import logging

from sklearn.feature_extraction.text import CountVectorizer

from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import Description, DescriptionList
from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import run_preprocessing_funcs, tf_idf, get_stopwords
from derive_conceptualspace.util.mpl_tools import show_hist
import math

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

class PPComponents():
    SKCOUNTVEC_SUPPORTS = ["sentwise_merge", "add_title", "add_subtitle", "remove_htmltags", "remove_stopwords", "convert_lower", "remove_diacritics", "add_additionals"]
    OPTION_LETTER = dict(
        sentwise_merge="m", #pre-preprocessing. If there are mutliple descriptions for different years, either add all unique sentences or calculate relative term-frequencies
        add_additionals="f",
        add_title="a",
        add_subtitle="u",
        remove_htmltags="h",
        sent_tokenize="t",
        convert_lower="c",
        remove_stopwords="s",
        lemmatize="l",
        remove_diacritics="d",
        remove_punctuation="p",
        use_skcountvec="2",
    )

    def __init__(self, **kwargs):
        assert kwargs.keys() == self.OPTION_LETTER.keys()
        if kwargs["use_skcountvec"]:
            must_override = [i for i in kwargs.keys()-self.SKCOUNTVEC_SUPPORTS-{"use_skcountvec"} if kwargs[i]]
            if must_override:
                print(f"Must overwrite the following PP-Components to False as SKLearn-CountVectorizer doesn't support it: {', '.join(must_override)}")
                raise Exception("No can do!")
                kwargs = {k: False if k in must_override else v for k, v in kwargs.items()}
        self.di = kwargs

    def __getattr__(self, item):
        assert item in self.di, f"{item} is no PP-Component!"
        return self.di[item]

    def __repr__(self):
        return "".join([v for k, v in self.OPTION_LETTER.items() if self.di[k]])

    @staticmethod
    def from_str(string):
        if str(string).lower() == "none":
            return PPComponents(**{k: False for k in PPComponents.OPTION_LETTER.keys()})
        tmp = PPComponents(**{k: v in str(string) for k, v in PPComponents.OPTION_LETTER.items()})
        assert str(tmp) == str(string), f"Is {str(tmp)} but should be {string}!"
        return tmp

########################################################################################################################
########################################################################################################################
########################################################################################################################

def preprocess_descriptions_full(raw_descriptions, dataset_class, pp_components, for_language, translate_policy, languages, translations=None):
    #TODO should I assert a minimal number of PP-Components? If I don't word-tokenize it all doesn't make much sense, does it?
    pp_components = PPComponents.from_str(pp_components)
    print("The following Pre-Processings will be performed:", ", ".join([k for k, v in pp_components.di.items() if v]))
    descriptions = dataset_class.preprocess_raw_file(raw_descriptions, pp_components)
    if get_setting("preprocessed_bow", default_false=True):
        descriptions = descriptions_from_bow(descriptions, languages, translations, translate_policy)
        if len(raw_descriptions["vecs"]) > len(descriptions):
            warnings.warn(f"Because of the min-words-per-desc setting, {len(raw_descriptions['vecs'])-len(descriptions)} of the original items needed to be removed!")
    else:
        if get_setting("DEBUG"):
            descriptions = descriptions[:get_setting("DEBUG_N_ITEMS")] #pd.DataFrame([descriptions.iloc[key] for key in random.sample(range(len(descriptions)), k=get_setting("DEBUG_N_ITEMS"))])
        if isinstance(languages, str):
            languages = {k: {k2:languages for k2 in descriptions[k]} if set(descriptions[k]) != {''} else None for k in descriptions.keys()}
        descriptions = create_bare_desclist(languages, translations, for_language, list(descriptions["title"]), list(descriptions["description"]),
                            [i if str(i) != "nan" else None for i in descriptions["subtitle"]], translate_policy, pp_components=pp_components, assert_all_translated=False,
                             additionals={i: [j if not (isinstance(j, float) and math.isnan(j)) else None for j in descriptions[i]] for i in dataset_class.additionals} if pp_components.add_additionals else None)
        if pp_components.use_skcountvec:
            descriptions = pp_descriptions_countvec(descriptions, pp_components, for_language)
        else:
            descriptions = preprocess_descriptions(descriptions, pp_components)
    descriptions = descriptions.filter_words(min_words=get_setting("MIN_WORDS_PER_DESC"))
    show_hist([i.n_words() for i in descriptions._descriptions], "Words per Description", xlabel="Number of Words")
    return descriptions, {"n_samples": len(descriptions)}


def descriptions_from_bow(descs, languages, translations, translate_policy):
    if translate_policy != "onlyorig" or languages != "en":
        raise NotImplementedError()
    desc_list = DescriptionList(add_title=False, add_subtitle=False, translate_policy=translate_policy, additionals_names=list(descs["classes"].keys()))
    if get_setting("DEBUG"):
        descs["vecs"] = dict(list(descs["vecs"].items())[:get_setting("DEBUG_N_ITEMS")])
    for name, bow in descs["vecs"].items():
        desc_list.add(Description(lang=languages, text=None, title=name, subtitle=None, orig_textlang=None, bow=bow,
                                  additionals={k: v.get(name) for k, v in descs["classes"].items()}))
    desc_list.proc_steps.append("bow")
    return desc_list


def create_bare_desclist(languages, translations, for_language, names, descriptions, subtitles, translate_policy, pp_components,
                         assert_all_translated=True, additionals=None):
    """Creates the Bare Descriptions-List. This function handles the *translate_policy* and the pp_components *add_coursetitle* and *add_subtitle*.
    All Other Preprocessing-steps must be done after this step. After this step the Raw Descriptions are not needed anymore."""
    additionals = additionals or {}
    desc_list = DescriptionList(add_title=pp_components.add_title, add_subtitle=pp_components.add_subtitle, translate_policy=translate_policy, additionals_names=list(additionals.keys()))

    if translate_policy == "origlang":
        for i in range(len(descriptions)):
            desc_list.add(Description(lang=languages["description"][descriptions[i]], orig_textlang=languages["description"][descriptions[i]],
                                      orig_titlelang=languages["title"][names[i]], orig_subtitlelang=(languages["subtitle"] or {}).get(subtitles[i]),
                                      text=descriptions[i], title=names[i], subtitle=subtitles[i], additionals={k: v[i] for k, v in additionals.items()}))
    elif translate_policy == "onlyorig":
        indices = [ind for ind, elem in enumerate([languages["description"][desc] for desc in descriptions]) if elem == for_language]
        print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the ones of language {for_language} ({len(indices)} left)")
        for i in indices:
            desc_list.add(Description(lang=for_language, text=descriptions[i], title=names[i], subtitle=subtitles[i], orig_textlang=for_language,
                                      orig_titlelang=languages["title"][names[i]] if languages["title"] is not None else "unk",
                                      orig_subtitlelang=(languages["subtitle"] or {}).get(subtitles[i]) if languages["subtitle"] is not None else "unk",
                                      additionals={k: v[i] for k, v in additionals.items()}))
    elif translate_policy == "translate":
        missing_translations = set()
        use_cols =                            {"description": ["text",     "orig_textlang",     "origlang_text"]}
        if pp_components.add_title: use_cols["title"] =       ["title",    "orig_titlelang",    "origlang_title"]
        if pp_components.add_subtitle: use_cols["subtitle"] = ["subtitle", "orig_subtitlelang", "origlang_subtitle"]
        for i, tmp in enumerate(zip(descriptions, names, subtitles)):
            di = {"description": tmp[0], "title": tmp[1], "subtitle": tmp[2]}
            kwargs = dict(origlang_text=di["description"], origlang_title=di["title"], lang=for_language, additionals={k: v[i] for k, v in additionals.items()})
            for col, components in use_cols.items():
                kwargs.update({components[2]: di[col]})
                if languages[col][di[col]] == for_language:
                    kwargs.update({components[0]: di[col], components[1]: for_language, components[2]: None})
                elif di[col] in translations[col]:
                    kwargs.update({components[0]: translations[col][di[col]], components[1]: languages[col][di[col]], components[2]: di[col]})
                else:
                    missing_translations.add(di["title"])
                    continue
            if di["title"] not in missing_translations:
                desc_list.add(Description(**kwargs))
        if len(desc_list) < len(names):
            print(f"Dropped {len(names) - len(desc_list)} out of {len(names)} descriptions because I will take ones for language {for_language} and ones with a translation")
        assert not (missing_translations and assert_all_translated)
    else:
        assert False, f"You specified a wrong translate_policy: {translate_policy}"
    desc_list.confirm("translate_policy", language=for_language)
    desc_list.languages = set(i.lang for i in desc_list._descriptions)
    return desc_list

########################################################################################################################
########################################################################################################################
########################################################################################################################

def preprocess_descriptions(descriptions, components):
    """3.4 in [DESC15]"""
    descriptions = run_preprocessing_funcs(descriptions, components)
    print("Ran the following preprocessing funcs:", ", ".join(descriptions.proc_steps))
    return descriptions



def get_countvec(pp_components, max_ngram, language, min_df=1):
    if isinstance(pp_components, str): pp_components=PPComponents.from_str(pp_components)
    if pp_components.remove_stopwords and get_setting("TRANSLATE_POLICY") == "origlang":
        raise NotImplementedError("Cannot deal with per-language-stopwords when using sklearn's CountVectorizer!")
    cnt = CountVectorizer(strip_accents="unicode" if pp_components.remove_diacritics else None,
                          lowercase = pp_components.convert_lower,
                          ngram_range=(1, max_ngram),
                          min_df=min_df, #If 2, every term has a "partner", making the dissimilarity-matrix more compact
                          stop_words=get_stopwords(language) if pp_components.remove_stopwords else None, #TODO see https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
                          )
    # I cannot set min_df and max_df, as I need all words for the dissimilarity-matrix!
    # TODO when I set preprocessor here I can override the preprocessing (strip_accents and lowercase) stage while preserving tokenizing and n-grams generation steps
    # TODO gucken wie viele Schritte mir das schon spart - keyword extraction, grundlage für dissim_matrix, ...? (Corollary: gucken was für min_df/max_df-ranges für dissim_matrix sinnvoll sind)
    # TODO I can merge this and the old one: If the PPComponents-Entry is uppercase, use a subcomponent of the countvectorizer instead of original one
    #  (it's both tokenization and occurence counting in one class, see https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage)
    return cnt


def pp_descriptions_countvec(descriptions, pp_components, for_language):
    if pp_components.remove_htmltags:
        descriptions.process_all(lambda data: re.compile(r'<.*?>').sub('', data), "remove_htmltags")
    cnt = get_countvec(pp_components, 1, for_language, min_df=1) #`build_analyzer`, in contrast to fit_transform, doesn't respect min_df anyway!!
    descriptions.process_all(cnt.build_preprocessor(), "preprocess")
    descriptions.process_all(cnt.build_tokenizer(), "tokenize")
    descriptions.process_all(cnt.build_analyzer(), "analyze", proc_base=lambda desc: " ".join(desc.processed_text))
    # TODO the analyze-step shouldn't find bigrams across sentences...! (resolves when sent_tokenizing)
    # TODO if I can do sent_tokenize here I also need to handle DescriptionList.generate_DocTermMatrix
    descriptions.recover_settings = dict(pp_components=str(pp_components), language=for_language)
    return descriptions
