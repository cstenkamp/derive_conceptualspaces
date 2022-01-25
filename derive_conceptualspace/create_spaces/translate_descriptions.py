from copy import deepcopy
from os.path import join, isfile, basename
import logging
import html

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.util.google_translate import translate_text
from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents

logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################

def get_langs(whatever_list, assert_len=True, pgbar_name=None):
    lans = {}
    for item in tqdm(whatever_list, desc=pgbar_name):
        try:
            lans[item] = detect(item)
        except LangDetectException as e:
            lans[item] = "unk"
    if assert_len:
        assert len(lans) == len(whatever_list)
    return lans


def translate_elems(whatever_list, languages=None, keys=None, already_translated=None):
    def val_to_key(val, return_all=False):
        #I need this shit only bc I had the coursetitles be the key for the coursedescription, that was dumb
        if keys is None:
            return val
        cands = [i for i in list(zip(keys, whatever_list)) if i[1] == val]
        assert len(set(i[1] for i in cands)) == 1
        if return_all:
            return [i[0] for i in cands]
        return cands[0][0]

    already_translated = already_translated or {}
    lst = dict(zip(whatever_list, languages))
    if already_translated:
        lst = {k: v for k, v in lst.items() if val_to_key(k) not in already_translated}
    untranslateds = [html.unescape(i) for i in lst.keys()]
    languages = list(lst.values())
    translateds = deepcopy(already_translated)

    print(f"There are {len(untranslateds)} items with {sum(len(i) for i in untranslateds)} characters to be translated.")
    if untranslateds:
        translations = translate_text(untranslateds, origlans=languages)
        new_ones = {val_to_key(k): v for k, v in zip(untranslateds, translations)}
        translateds.update(new_ones)
        print(f"You translated {len('.'.join(untranslateds[:len(new_ones)]))} (becoming {len('.'.join(new_ones.values()))}) Chars from {len(new_ones)} items.")
        n_untranslated = len(set(untranslateds))-len(set(new_ones.keys()))
        print(f"Now, {len(translateds)}/{n_untranslated+len(translateds)} ({len(translateds)/(n_untranslated+len(translateds))*100:.1f}%) items are translated")
    else:
        n_untranslated = 0
    if keys is not None:
        alternative_keys = {i: tmp for i in whatever_list if len((tmp := val_to_key(i, return_all=True))) > 1}
        missing_keys = {k: [i for i in v if i not in translateds] for k,v in alternative_keys.items() if not all(i in translateds for i in v)}
        if missing_keys:
            other_keys = {k: [i for i in v if i in translateds][0] for k,v in alternative_keys.items() if any(i in translateds for i in v)}
            stolen_translations = {i: translateds[other_keys[i]] for i in {k:i for k, v in missing_keys.items() for i in v}}
            stolen_trsl_keys = {i: stolen_translations[k] for k, v in missing_keys.items() for i in v }
            translateds.update(stolen_trsl_keys)
    return translateds, True, n_untranslated==0  #translateds, did_update, is_complete



def create_languages_file(raw_descriptions, columns, json_persister, dataset_class, declare_silent=False, pp_components=None, proc_descs=None):
    if isinstance(columns, str):
        columns = [columns]
    results = {}
    for col in columns:
        try:
            languages = json_persister.load(None, f"{col}_languages", loader=lambda langs: langs, silent=declare_silent)
        except FileNotFoundError:
            if proc_descs is None:
                proc_descs = dataset_class.preprocess_raw_file(raw_descriptions, pp_components=PPComponents.from_str(pp_components))
            langs = get_langs(proc_descs[col], assert_len=False, pgbar_name=f"Getting Language of {col}")
            langs = {i[col]: langs[i[col]] for _,i in proc_descs.iterrows()}
            json_persister.save(f"{col}_languages.json", langs=langs, ignore_confs=["DEBUG", "PP_COMPONENTS", "TRANSLATE_POLICY", "LANGUAGE"])
            languages = json_persister.load(None, f"{col}_languages", loader=lambda langs: langs, silent=declare_silent)
        else:
            print(f"Languages-file for {col} already exists!")
        results[col] = languages
    return results


def full_translate_column(raw_descriptions, translate_policy, language, column, json_persister, dataset_class, pp_components=None):
    pp_components = PPComponents.from_str(pp_components)
    if translate_policy == "translate" and ((column == "description") or (column == "title" and pp_components.add_title) or (column == "subtitle" and pp_components.add_subtitle)):
        try:
            translations = json_persister.load(None, f"{column}_translations", silent=True)
        except FileNotFoundError:
            translations = dict(translations={}, is_complete=False)
        if "is_complete" not in translations: #for backwards compatibility
            translations = dict(translations=translations, is_complete=False)
        if not translations["is_complete"]:
            descriptions = dataset_class.preprocess_raw_file(raw_descriptions, pp_components=pp_components)
            languages = create_languages_file(raw_descriptions, column, json_persister, dataset_class, proc_descs=descriptions)[column]
            to_translate = {i: languages[i] for i in descriptions[column] if languages[i] != language and i not in translations["translations"]}
            translateds, did_update, is_complete = translate_elems(list(to_translate.keys()), list(to_translate.values()), already_translated=translations["translations"])
            if did_update:
                json_persister.save(f"{column}_translations.json", translations=translateds, is_complete=is_complete, force_overwrite=True, ignore_confs=["DEBUG", "PP_COMPONENTS", "TRANSLATE_POLICY"])
            translations = json_persister.load(None, f"{column}_translations", silent=True)
            if not is_complete:
                print(f"The translated {column}s are not complete yet!")
                exit(1)
            else:
                print(f"The translated {column}s are now complete! Yay!")
        return translations
