from os.path import basename, isfile
import re
import logging
import random

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import Description
from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import run_preprocessing_funcs, make_bow, tf_idf

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


class PPComponents():
    def __init__(self, add_coursetitle=True, add_subtitle=True, count_vec=False,
                 sent_tokenize=True, convert_lower=True, remove_stopwords=True, lemmatize=True, remove_diacritics=True, remove_punctuation=True):
        if count_vec:
            self.di = dict(add_coursetitle=add_coursetitle, add_subtitle=add_subtitle, count_vec=True)
        else:
            self.di = dict(add_coursetitle=add_coursetitle, add_subtitle=add_subtitle, sent_tokenize=sent_tokenize, convert_lower=convert_lower,
                       remove_stopwords=remove_stopwords, lemmatize=lemmatize, remove_diacritics=remove_diacritics, remove_punctuation=remove_punctuation)

    def get(self, item, default=None):
        return self.di.get(item, default)

    def __getattr__(self, item):
        assert item in self.di
        return self.get(item, False)

    def __getitem__(self, item):
        return self.__getattr__(item)


    def __repr__(self):
        return (
            ("a" if self["add_coursetitle"] else "") + ("u" if self["add_subtitle"] else "") +
            ("2" if self["count_vec"] else #yes, this is on purpose, if using cont_vec the rest is not applied! (#TODO parts of this can, see CountVectorizer docs)
                ("t" if self["sent_tokenize"] else "") + ("c" if self["convert_lower"] else "") +
                ("s" if self["remove_stopwords"] else "") + ("l" if self["lemmatize"] else "") +
                ("d" if self["remove_diacritics"] else "") + ("p" if self["remove_punctuation"] else "")
            )
        )

    @staticmethod
    def from_str(string):
        if "2" in string:
            return PPComponents(add_coursetitle="a" in string, add_subtitle="u" in string, count_vec=True)
        return PPComponents(add_coursetitle="a" in string, add_subtitle="u" in string, sent_tokenize="t" in string, convert_lower="c" in string,
                            remove_stopwords="s" in string, lemmatize="l" in string, remove_diacritics="d" in string, remove_punctuation="p" in string)



def preprocess_descriptions_full(raw_descriptions, pp_components, translate_policy, languages, translations, title_languages, title_translations):
    #TODO options to consider language, fachbereich, and to add [translated] title to description
    pp_components = PPComponents.from_str(pp_components)
    descriptions = preprocess_raw_course_file(raw_descriptions)
    if get_setting("DEBUG"):
        descriptions = pd.DataFrame([descriptions.iloc[key] for key in random.sample(range(len(descriptions)), k=get_setting("DEBUG_N_ITEMS"))])
    descriptions = handle_translations(languages, translations, list(descriptions["Name"]), list(descriptions["Beschreibung"]),
                                       [i if str(i) != "nan" else None for i in descriptions["Untertitel"]], translate_policy, title_languages, title_translations,
                                       add_coursetitle=pp_components["add_coursetitle"], add_subtitle=pp_components["add_subtitle"], assert_all_translated=True) #TODO only if translate?!
    if pp_components.count_vec:
        vocab, descriptions = pp_descriptions_countvec(descriptions, pp_components)
    else:
        vocab, descriptions = preprocess_descriptions(descriptions, pp_components)
    return vocab, descriptions


def pp_descriptions_countvec(descriptions, pp_components):
    #TODO make stuff like ngram-range parameters, and play around with values for the many options this has!!
    cnt = CountVectorizer(strip_accents="unicode", ngram_range=(1, 3), max_df=0.9, min_df=10)
    # TODO when I set preprocessor here I can override the preprocessing (strip_accents and lowercase) stage while preserving tokenizing and n-grams generation steps
    # TODO gucken wie viele Schritte mir das schon spart - keyword extraction, grundlage für dissim_matrix, ...? (Corollary: gucken was für min_df/max_df-ranges für dissim_matrix sinnvoll sind)
    # TODO I can merge this and the old one: If the PPComponents-Entry is uppercase, use a subcomponent of the countvectorizer instead of original one
    #  (it's both tokenization and occurence counting in one class, see https://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage)
    for comp, name in (cnt.build_preprocessor(), "preprocess"), (cnt.build_tokenizer(), "tokenize"), (lambda x: cnt.build_analyzer()(" ".join(x)), "analyze"):
        for desc in descriptions:
            desc.process(comp(desc.processed_text), name) #TODO the analyze-step shouldn't find bigrams across sentences...! (resolves when sent_tokenizing)
    X = cnt.fit_transform([i.unprocessed_text for i in descriptions])
    aslist = [list(sorted(zip((tmp := X.getrow(nrow).tocoo()).col, tmp.data), key=lambda x:x[0])) for nrow in range(X.shape[0])]
    all_words = {v: k for k, v in cnt.vocabulary_.items()}
    if False:
        #`make_bow` adds the complete counter, including those words which shouldn't be counted due to max_df and min_df values, so we don't do it
        vocab, descriptions = make_bow(descriptions)
        for i in random.sample(range(len(descriptions)), 20):
            assert all(j in descriptions[i] for j in [all_words[w] for w,n in aslist[i]])
            assert {all_words[j[0]]: j[1] for j in aslist[i]}.items() <= dict(descriptions[i].bow).items()
    else:
        for j, desc in enumerate(descriptions):
            desc.bow = {all_words[i[0]]: i[1] for i in aslist[j]}
        all_words = list(sorted(cnt.vocabulary_.keys()))
    return all_words, descriptions



def preprocess_raw_course_file(df, min_desc_len=10):
    """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
        dropping duplicates"""
    #TODO in exploration I also played around with Levenhsthein-distance etc!
    #remove those for which the Name (exluding stuff in parantheses) is equal...
    df['NameNoParanth'] = df['Name'].str.replace(re.compile(r'\([^)]*\)'), '')
    df = df.drop_duplicates(subset='NameNoParanth')
    #remove those with too short a description...
    df = df[~df['Beschreibung'].isna()]
    df.loc[:, 'desc_len'] = [len(i) for i in df['Beschreibung']]
    df = df[df["desc_len"] > min_desc_len]
    df = df.drop(columns=['desc_len','NameNoParanth'])
    #remove those with equal Veranstaltungsnummer...
    df = df.drop_duplicates(subset='VeranstaltungsNummer')
    #TODO instead of dropping, I could append descriptions or smth!!
    for column in ["Name", "Beschreibung", "Untertitel"]:
        df[column] = df[column].str.strip()
    if len((dups := df[df["Name"].duplicated(keep=False)])) > 0:
        print("There are courses with different VeranstaltungsNummer but equal Name!")
        for n, cont in dups.iterrows():
            df.loc[n]["Name"] = f"{cont['Name']} ({cont['VeranstaltungsNummer']})"
    return df


def handle_translations(languages, translations, names, descriptions, subtitles, translate_policy, title_languages, title_translations,
                        assert_all_translated=True, add_coursetitle=False, add_subtitle=False):
    orig_lans = [languages[i] for i in names]
    if translate_policy == "origlan":
        result = [Description(add_title=add_coursetitle, add_subtitle=add_subtitle, lang=orig_lans[i], text=descriptions[i],
                              title=names[i], subtitle=subtitles[i], orig_textlang=orig_lans[i], orig_titlelang=title_languages[names[i]],
                             ) for i in range(len(descriptions))]
    elif translate_policy == "onlyeng":
        raise NotImplementedError("TODO: add_coursetitle und add_subtitle for this")
        # indices = [ind for ind, elem in enumerate(orig_lans) if elem == "en"]
        # print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones")
        # names, descriptions = [names[i] for i in indices], [descriptions[i] for i in indices]
        # result = [Description(text=descriptions[i], lang="en", for_name=names[i], orig_lang="en") for i in range(len(descriptions))]
    elif translate_policy == "translate":
        result = []
        missing_translations = set()
        for desc, title, subtitle in zip(descriptions, names, subtitles):
            kwargs = dict(add_title=add_coursetitle, add_subtitle=add_subtitle, origlang_text=desc, lang="en") # title=title, subtitle=subtitle
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
            # missing: subtitle, orig_titlelang, origlang_title
            result.append(Description(**kwargs))
        if len(result) < len(names):
            print(f"Dropped {len(names) - len(result)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
        assert not (missing_translations and assert_all_translated)
        assert all((i.is_translated and i.lang == "en" and i.orig_textlang != "en") or (not i.is_translated and i.orig_textlang == i.lang and i.orig_textlang == "en") for i in result)
    else:
        assert False
    return result


def preprocess_descriptions(descriptions, components):
    """3.4 in [DESC15]"""
    # TODO there's https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # TODO it must save which kind of preprocessing it did (removed stop words, convered lowercase, stemmed, ...)
    descriptions = run_preprocessing_funcs(descriptions, components) #TODO allow the CountVectorizer!
    print("Ran the following preprocessing funcs:", ", ".join([i[1] for i in descriptions[0].processing_steps]))
    vocab, descriptions = make_bow(descriptions)
    return vocab, descriptions
