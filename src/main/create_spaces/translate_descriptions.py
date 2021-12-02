from os.path import join, isfile, basename
import logging
import os
import json
import html

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from src.static.settings import SID_DATA_BASE
from src.main.util.pretty_print import pretty_print as print
from src.main.util.google_translate import translate_text
from src.main.util.mds_object import ORIGLAN

logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_load_languages_file(names, descriptions, file_path=SID_DATA_BASE, filename="languages.json"):
    if not isfile(join(file_path, filename)):
        lans = []
        print("Finding languages of descriptions...")
        for desc in tqdm(descriptions):
            try:
                lans.append(detect(desc))
            except LangDetectException as e:
                lans.append("unk")
        lans = dict(zip(names, lans))
        with open(join(file_path, filename), "w") as ofile:
            json.dump(lans, ofile)
    else:
        with open(join(file_path, filename), "r") as rfile:
            lans = json.load(rfile)
    return lans


def translate_descriptions(base_dir, mds_basename):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith(mds_basename) and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=ORIGLAN)
    names, descriptions, mds, languages = mds_obj.names, mds_obj.descriptions, mds_obj.mds, mds_obj.languages
    #TODO use languages
    assert len(set(names)) == len(names)
    descriptions = [html.unescape(i) for i in descriptions]
    name_desc = dict(zip(names, descriptions))
    if isfile((translationsfile := join(base_dir, "translated_descriptions.json"))):
        with open(translationsfile, "r") as rfile:
            translateds = json.load(rfile)
    else:
        translateds = {}
    languages = create_load_languages_file(names, descriptions)
    untranslated = {k:v for k,v in name_desc.items() if languages[k] != "en" and k not in translateds}
    if not untranslated:
        print("Everything is translated!!")
        return
    print(f"There are {len(''.join([i[0] for i in untranslated.values()]))} descriptions to be translated.")
    translations = translate_text([name_desc[k] for k in untranslated], origlans=[languages[k] for k in untranslated])
    # hash_translates = dict(zip([hashlib.sha256(i.encode("UTF-8")).hexdigest() for i in to_translate], translations))
    translateds.update(dict(zip(untranslated.keys(), translations)))
    with open(join(SID_DATA_BASE, "translated_descriptions.json"), "w") as wfile:
        json.dump(translateds, wfile)
    translation_new_len = len("".join(translations))
    translated_descs = [name_desc[i] for i in name_desc.keys() if i in set(dict(zip([k for k in untranslated], translations)).keys())]
    print(f"You translated {len('.'.join(translated_descs))} (becoming {translation_new_len}) Chars from {len(translated_descs)} descriptions.")


def count_translations(base_dir, mds_basename):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith(mds_basename) and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=ORIGLAN)
    names, descriptions, mds, languages = mds_obj.names, mds_obj.descriptions, mds_obj.mds, mds_obj.languages
    assert len(set(names)) == len(names)
    name_desc = dict(zip(names, descriptions))
    if isfile((translationsfile := join(base_dir, "translated_descriptions.json"))):
        with open(translationsfile, "r") as rfile:
            translateds = json.load(rfile)
    else:
        translateds = {}
    languages = create_load_languages_file(names, descriptions)
    all_untranslateds = {k: v for k, v in name_desc.items() if languages[k] != "en" and k not in translateds}
    all_translateds = {k: v for k, v in name_desc.items() if languages[k] != "en" and k in translateds}
    all_notranslateds = {k: v for k, v in name_desc.items() if languages[k] == "en"}

    print("Regarding #Descriptions:")
    print(f"{len(all_untranslateds)+len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)} ({round((len(all_untranslateds)+len(all_translateds))/(len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)), 4)*100}%) need to be translated")
    print(f"{len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)} ({round((len(all_translateds))/(len(all_untranslateds)+len(all_translateds)), 4)*100}%) are translated")
    print("Reagarding #Chars:")
    all_untranslateds, all_translateds, all_notranslateds = "".join(list(all_untranslateds.values())), "".join(list(all_translateds.values())), "".join(list(all_notranslateds.values()))
    print(f"{len(all_untranslateds)+len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)} ({round((len(all_untranslateds)+len(all_translateds))/(len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)), 4)*100}%) need to be translated")
    print(f"{len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)} ({round((len(all_translateds))/(len(all_untranslateds)+len(all_translateds)), 4)*100}%) are translated")
