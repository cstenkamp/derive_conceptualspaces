from os.path import basename, isfile, join, dirname
import re
import logging
import random


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import Description, DescriptionList
from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import run_preprocessing_funcs, tf_idf, get_stopwords
from derive_conceptualspace.util.mpl_tools import show_hist

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

class PPComponents():
    SKCOUNTVEC_SUPPORTS = ["add_coursetitle", "add_subtitle", "remove_htmltags", "remove_stopwords", "convert_lower", "remove_diacritics", "add_fachbereich"]
    OPTION_LETTER = dict(
        add_fachbereich="f", #TODO generalize this for other datasets
        add_coursetitle="a", #TODO generalize this for other datasets
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
        assert item in self.di
        return self.di[item]

    def __repr__(self):
        return "".join([v for k, v in self.OPTION_LETTER.items() if self.di[k]])

    @staticmethod
    def from_str(string):
        tmp = PPComponents(**{k: v in string for k, v in PPComponents.OPTION_LETTER.items()})
        assert str(tmp) == string, f"Is {str(tmp)} but should be {string}!"
        return tmp

########################################################################################################################
########################################################################################################################
########################################################################################################################

def preprocess_descriptions_full(raw_descriptions, dataset_specifics_module, pp_components, translate_policy, languages, translations=None, title_languages=None, title_translations=None):
    #TODO options to consider language and fachbereich
    max_ngram = get_setting("MAX_NGRAM") if get_setting("NGRAMS_IN_EMBEDDING") else 1
    pp_components = PPComponents.from_str(pp_components)
    descriptions = dataset_specifics_module.preprocess_raw_file(raw_descriptions)
    if get_setting("DEBUG"):
        descriptions = pd.DataFrame([descriptions.iloc[key] for key in random.sample(range(len(descriptions)), k=get_setting("DEBUG_N_ITEMS"))])
    descriptions = create_bare_desclist(languages, translations, list(descriptions["Name"]), list(descriptions["Beschreibung"]),
                        [i if str(i) != "nan" else None for i in descriptions["Untertitel"]], translate_policy, title_languages, title_translations,
                        add_coursetitle=pp_components.add_coursetitle, add_subtitle=pp_components.add_subtitle, assert_all_translated=True, #TODO only if translate?!
                        additionals=list(descriptions["Fachbereich"]) if pp_components.add_fachbereich else None, additionals_names=["Fachbereich"] if pp_components.add_fachbereich else None)
    if pp_components.use_skcountvec:
        descriptions = pp_descriptions_countvec(descriptions, pp_components, max_ngram).filter_words(min_words=get_setting("MIN_WORDS_PER_DESC"))
        metainf = {"n_samples": len(descriptions), "ngrams_in_embedding": get_setting("NGRAMS_IN_EMBEDDING"), **({"pp_max_ngram": get_setting("MAX_NGRAM")} if get_setting("NGRAMS_IN_EMBEDDING") else {}),
                   "min_words": get_setting("MIN_WORDS_PER_DESC")}
    else:
        assert max_ngram == 1, "Cannot deal with n-grams without SKLearn!"
        descriptions = preprocess_descriptions(descriptions, pp_components).filter_words(min_words=get_setting("MIN_WORDS_PER_DESC"))
        metainf = {"n_samples": len(descriptions), "ngrams_in_embedding": False, "min_words": get_setting("MIN_WORDS_PER_DESC")}
    show_hist([i.n_words() for i in descriptions._descriptions], "Words per Description", xlabel="Number of Words")
    return descriptions, metainf



def create_bare_desclist(languages, translations, names, descriptions, subtitles, translate_policy, title_languages, title_translations,
                        assert_all_translated=True, add_coursetitle=False, add_subtitle=False, additionals=None, additionals_names=None):
    """Creates the Bare Descriptions-List. This function handles the *translate_policy* and the pp_components *add_coursetitle* and *add_subtitle*.
    All Other Preprocessing-steps must be done after this step. After this step the Raw Descriptions are superflous."""
    title_languages = title_languages or languages
    orig_lans = [languages[i] for i in names]
    desc_list = DescriptionList(add_title=add_coursetitle, add_subtitle=add_subtitle, translate_policy=translate_policy, additionals_names=additionals_names)

    if translate_policy == "origlang":
        for i in range(len(descriptions)):
            desc_list.add(Description(lang=orig_lans[i], text=descriptions[i], title=names[i], subtitle=subtitles[i], orig_textlang=orig_lans[i],
                                      orig_titlelang=title_languages[names[i]], additionals=additionals[i] if additionals is not None else None
                                      ))
    elif translate_policy == "onlyeng":
        indices = [ind for ind, elem in enumerate(orig_lans) if elem == "en"]
        print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones ({len(indices)} left)")
        for i in indices:
            desc_list.add(Description(lang="en", text=descriptions[i], title=names[i], subtitle=subtitles[i], orig_textlang="en",
                                      orig_titlelang=title_languages[names[i]], additionals=additionals[i] if additionals is not None else None
                                      ))
    elif translate_policy == "translate":
        missing_translations = set()
        for i, (desc, title, subtitle) in enumerate(zip(descriptions, names, subtitles)):
            kwargs = dict(origlang_text=desc, lang="en", additionals=additionals[i] if additionals is not None else None)
            if languages[title] == "en" or title in translations:
                if languages[title] == "en":
                    kwargs.update(dict(text=desc, orig_textlang="en", origlang_text=None))
                else:
                    kwargs.update(dict(text=translations[title], orig_textlang=languages[title], origlang_text=desc))
            else:
                missing_translations.add(title)
                continue
            if title_languages[title] == "en" or title in title_translations:
                if title_languages[title] == "en":
                    kwargs.update(dict(orig_titlelang="en", title=title, subtitle=subtitle))
                else:
                    assert subtitle is None or subtitle in title_translations
                    kwargs.update(dict(orig_titlelang=title_languages[title], origlang_title=title, title=title_translations[title], subtitle=title_translations.get(subtitle), origlang_subtitle=subtitle))
            else:
                missing_translations.add(title)
                continue
            desc_list.add(Description(**kwargs))
        if len(desc_list) < len(names):
            print(f"Dropped {len(names) - len(desc_list)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
        assert not (missing_translations and assert_all_translated)
    else:
        assert False, f"You specified a wrong translate_policy: {translate_policy}"
    desc_list.confirm("translate_policy")
    return desc_list

########################################################################################################################
########################################################################################################################
########################################################################################################################

def preprocess_descriptions(descriptions, components):
    """3.4 in [DESC15]"""
    # TODO there's https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # TODO it must save which kind of preprocessing it did (removed stop words, convered lowercase, stemmed, ...)
    descriptions = run_preprocessing_funcs(descriptions, components) #TODO allow the CountVectorizer!
    print("Ran the following preprocessing funcs:", ", ".join(descriptions.proc_steps))
    descriptions.proc_min_df = 1
    descriptions.proc_ngram_range = (1, 1)
    # vocab, descriptions = make_bow(descriptions) #TODO correct to remove?!
    # vocab = sorted(set(flatten([flatten(desc.processed_text) for desc in descriptions])))
    # return vocab, descriptions
    return descriptions



def get_countvec(pp_components, max_ngram, min_df=1):
    #TODO play around with values for the many options this has!!
    if pp_components.remove_stopwords and get_setting("TRANSLATE_POLICY") == "origlang":
        raise NotImplementedError("Cannot deal with per-language-stopwords in this mode")
    cnt = CountVectorizer(strip_accents="unicode" if pp_components.remove_diacritics else None,
                          lowercase = pp_components.convert_lower,
                          ngram_range=(1, max_ngram),
                          min_df=min_df, #If 2, every component of a description has a "partner", I THINK that makes the dissimilarity-matrix better
                          stop_words=get_stopwords("en") if pp_components.remove_stopwords else None, #TODO see https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
                          )
    # I cannot set min_df and max_df, as I need all words for the dissimilarity-matrix!
    # TODO when I set preprocessor here I can override the preprocessing (strip_accents and lowercase) stage while preserving tokenizing and n-grams generation steps
    # TODO gucken wie viele Schritte mir das schon spart - keyword extraction, grundlage für dissim_matrix, ...? (Corollary: gucken was für min_df/max_df-ranges für dissim_matrix sinnvoll sind)
    # TODO I can merge this and the old one: If the PPComponents-Entry is uppercase, use a subcomponent of the countvectorizer instead of original one
    #  (it's both tokenization and occurence counting in one class, see https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage)
    return cnt


def pp_descriptions_countvec(descriptions, pp_components, max_ngram):
    if pp_components.remove_htmltags:
        descriptions.process_all(lambda data: re.compile(r'<.*?>').sub('', data), "remove_htmltags")
    cnt = get_countvec(pp_components, max_ngram, min_df=1) #as the build_analyzer, in contrast to fit_transform, doesn't respect min_df anyway!!
    descriptions.process_all(cnt.build_preprocessor(), "preprocess")
    descriptions.process_all(cnt.build_tokenizer(), "tokenize")
    descriptions.process_all(cnt.build_analyzer(), "analyze", adds_ngrams=max_ngram>1, proc_base=lambda desc: " ".join(desc.processed_text))
    # TODO the analyze-step shouldn't find bigrams across sentences...! (resolves when sent_tokenizing)
    descriptions.proc_min_df = 1
    descriptions.proc_ngram_range = cnt.ngram_range
    descriptions.recover_settings = dict(pp_components=str(pp_components), max_ngram=max_ngram)
    return descriptions



    # X = cnt.fit_transform([i.unprocessed_text for i in descriptions])
    # # aslist = [list(sorted(zip((tmp := X.getrow(nrow).tocoo()).col, tmp.data), key=lambda x:x[0])) for nrow in range(X.shape[0])]
    # all_words = {v: k for k, v in cnt.vocabulary_.items()}
    # # if False:
    # #     #`make_bow` adds the complete counter, including those words which shouldn't be counted due to max_df and min_df values, so we don't do it
    # #     vocab, descriptions = make_bow(descriptions)
    # #     for i in random.sample(range(len(descriptions)), 20):
    # #         assert all(j in descriptions[i] for j in [all_words[w] for w,n in aslist[i]])
    # #         assert {all_words[j[0]]: j[1] for j in aslist[i]}.items() <= dict(descriptions[i].bow).items()
    # # else:
    # #     for j, desc in enumerate(descriptions):
    # #         desc.bow = {all_words[i[0]]: i[1] for i in aslist[j]}
    # #     all_words = list(sorted(cnt.vocabulary_.keys()))
    # # TODO I removed this because it doesn't make sense to set the bag-of-words already here bc this may not contain n-grams, but in the extract-candidates we want to get those.
    # return all_words, descriptions
    # #TODO the CountVectorizer mixes up steps 1 and 2 - it prepares which n-grams to be able to extract later already here!

