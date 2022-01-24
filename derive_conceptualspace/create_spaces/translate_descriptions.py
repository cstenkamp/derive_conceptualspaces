from copy import deepcopy
from os.path import join, isfile, basename
import logging
import html

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from derive_conceptualspace.settings import get_setting
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.util.google_translate import translate_text
from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents

logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################

def get_langs(whatever_list, assert_len=True):
    lans = {}
    for item in tqdm(whatever_list):
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



def full_translate_titles(raw_descriptions, pp_components, translate_policy, title_languages_file, title_translations_file, json_persister, dataset_class):
    pp_components = PPComponents.from_str(pp_components)
    if (pp_components.add_coursetitle or pp_components.add_subtitle) and translate_policy == "translate": #TODO parts of this also needs to be done for onlyeng
        title_languages = create_languages_file(title_languages_file, "title_languages", "Name", json_persister, raw_descriptions, dataset_class)
        try:
            title_translations = json_persister.load(title_translations_file, "title_translations", silent=True)
        except FileNotFoundError:
            title_translations = dict(title_translations={}, is_complete=False)
        if not title_translations.get("is_complete", False):
            descriptions = dataset_class.preprocess_raw_file(raw_descriptions)
            subtitles = [i if str(i) != "nan" else None for i in descriptions["Untertitel"]]
            to_translate = list(zip(*[(k, v) for k, v in title_languages.items() if v != "en"]))
            sec_translate = list(zip(*[i for i in zip(subtitles, title_languages.values()) if i[0] and i[1] != "en"]))
            translateds, did_update, is_complete= translate_elems(to_translate[0]+sec_translate[0], to_translate[1]+sec_translate[1], already_translated=title_translations["title_translations"] if "title_translations" in title_translations else title_translations)
            if did_update:
                json_persister.save(title_translations_file, title_translations=translateds, is_complete=is_complete, force_overwrite=True, ignore_confs=["DEBUG", "PP_COMPONENTS", "TRANSLATE_POLICY"])
            title_translations = json_persister.load(title_translations_file, "title_translations", silent=True)
            if not is_complete:
                print("The translated Titles are not complete yet!")
                exit(1)
            else:
                print("The translated titles are now complete! Yay!")

        return title_translations, title_languages
    return None, None


def create_languages_file(languages_file, file_basename, column, json_persister, raw_descriptions, dataset_class, declare_silent=False):
    try:
        languages = json_persister.load(languages_file, file_basename, loader=lambda langs: langs, silent=declare_silent)
    except FileNotFoundError:
        descriptions = dataset_class.preprocess_raw_file(raw_descriptions)
        langs = get_langs(descriptions[column], assert_len=False)
        langs = {i["Name"]: langs[i[column]] for _,i in descriptions.iterrows()}
        json_persister.save(basename(languages_file), langs=langs, ignore_confs=["DEBUG"])
        languages = json_persister.load(languages_file, file_basename, loader=lambda langs: langs, silent=declare_silent)
    return languages




def full_translate_descriptions(raw_descriptions, translate_policy, languages_file, translations_file, json_persister, dataset_class):
    if translate_policy == "translate":
        languages = create_languages_file(languages_file, "languages", "Beschreibung", json_persister, raw_descriptions, dataset_class)
        try:
            translations = json_persister.load(translations_file, "translations", silent=True)
        except FileNotFoundError:
            translations = dict(translations={}, is_complete=False)
        if "is_complete" not in translations: #for backwards compatibility
            translations = dict(translations=translations, is_complete=False)
        if not translations["is_complete"]:
            descriptions = dataset_class.preprocess_raw_file(raw_descriptions)
            assert len(descriptions) == len(languages)
            to_translate = dict([i for i in zip(descriptions["Beschreibung"], languages.values()) if i[1] != "en" and i[0] not in translations["translations"]])
            translateds, did_update, is_complete = translate_elems(list(to_translate.keys()), list(to_translate.values()), already_translated=translations["translations"])
            if did_update:
                json_persister.save(translations_file, translations=translateds, is_complete=is_complete, force_overwrite=True, ignore_confs=["DEBUG", "PP_COMPONENTS", "TRANSLATE_POLICY"])
            translations = json_persister.load(translations_file, "translations", silent=True)
            if not is_complete:
                print("The translated Descriptions are not complete yet!")
                exit(1)
            else:
                print("The translated Descriptions are now complete! Yay!")

        return translations, languages
    return None, None
