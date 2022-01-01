from os.path import basename, isfile
import re
import logging
import random

import pandas as pd

from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import Description
from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import run_preprocessing_funcs, make_bow, tf_idf

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


class PPComponents():
    def __init__(self, add_coursetitle=True, add_subtitle=True, sent_tokenize=True, convert_lower=True,
                 remove_stopwords=True, lemmatize=True, remove_diacritics=True, remove_punctuation=True):
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
            ("t" if self["sent_tokenize"] else "") + ("c" if self["convert_lower"] else "") +
            ("s" if self["remove_stopwords"] else "") + ("l" if self["lemmatize"] else "") +
            ("d" if self["remove_diacritics"] else "") + ("p" if self["remove_punctuation"] else "")
        )

    @staticmethod
    def from_str(string):
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
    vocab, descriptions = preprocess_descriptions(descriptions, pp_components)
    return vocab, descriptions



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
