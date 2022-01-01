from copy import deepcopy
from os.path import join, isfile, basename
import logging
import html

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.util.google_translate import translate_text
from derive_conceptualspace.create_spaces.preprocess_descriptions import preprocess_raw_course_file, PPComponents

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



def full_translate_titles(raw_descriptions, pp_components, translate_policy, title_languages_file, title_translations_file, json_persister):
    pp_components = PPComponents.from_str(pp_components)
    if (pp_components["add_coursetitle"] or pp_components["add_subtitle"]) and translate_policy == "translate": #TODO parts of this also needs to be done for onlyeng
        try:
            title_languages = json_persister.load(title_languages_file, "title_languages", ignore_params=["pp_components", "translate_policy"], loader=lambda title_langs: title_langs)
        except FileNotFoundError:
            descriptions = preprocess_raw_course_file(raw_descriptions)
            full_titles = [(tit+". \n"+sub) if str(sub) != "nan" else tit for tit,sub in zip(descriptions["Name"], descriptions["Untertitel"])]
            title_langs = dict(zip(list(descriptions["Name"]), get_langs(full_titles).values()))
            json_persister.save(title_languages_file, title_langs=title_langs, relevant_params=[])
            title_languages = json_persister.load(title_languages_file, "title_languages", ignore_params=["pp_components", "translate_policy"], loader=lambda title_langs: title_langs)

        try:
            title_translations = json_persister.load(title_translations_file, "translated_titles", ignore_params=["pp_components", "translate_policy"], force_overwrite=True)
        except FileNotFoundError:
            title_translations = dict(title_translations={}, is_complete=False)
        if not title_translations["is_complete"]:
            descriptions = preprocess_raw_course_file(raw_descriptions)
            subtitles = [i if str(i) != "nan" else None for i in descriptions["Untertitel"]]
            to_translate = list(zip(*[(k, v) for k, v in title_languages.items() if v != "en"]))
            sec_translate = list(zip(*[i for i in zip(subtitles, title_languages.values()) if i[0] and i[1] != "en"]))
            translateds, did_update, is_complete= translate_elems(to_translate[0]+sec_translate[0], to_translate[1]+sec_translate[1], already_translated=title_translations["title_translations"])
            if did_update:
                json_persister.save(title_translations_file, title_translations=translateds, is_complete=is_complete, relevant_params=[], force_overwrite=True)
            title_translations = json_persister.load(title_translations_file, "translated_titles", ignore_params=["pp_components", "translate_policy"], force_overwrite=True)
            if not is_complete:
                print("The translated Titles are not complete yet!")
                exit(1)
            else:
                print("The translated titles are now complete! Yay!")

        return title_translations, title_languages
    return None, None




def full_translate_descriptions(raw_descriptions, translate_policy, languages_file, translations_file, json_persister):
    if translate_policy == "translate": #TODO parts of this also needs to be done for onlyeng
        try:
            languages = json_persister.load(languages_file, "languages", ignore_params=["translate_policy"], loader=lambda langs: langs)
        except FileNotFoundError:
            descriptions = preprocess_raw_course_file(raw_descriptions)
            langs = get_langs(descriptions["Beschreibung"], assert_len=False)
            langs = {i["Name"]: langs[i["Beschreibung"]] for _,i in descriptions.iterrows()}
            json_persister.save(languages_file, langs=langs, relevant_params=[])
            languages = json_persister.load(languages_file, "languages", ignore_params=["translate_policy"], loader=lambda langs: langs)

        try:
            translations = json_persister.load(translations_file, "translated_descriptions", ignore_params=["translate_policy"], force_overwrite=True)
        except FileNotFoundError:
            translations = dict(translations={}, is_complete=False)
        if "is_complete" not in translations: #for backwards compatibility
            json_persister.save(translations_file, translations=translations, is_complete=False, relevant_params=[], force_overwrite=True)
            translations = json_persister.load(translations_file, "translated_descriptions", ignore_params=["translate_policy"], force_overwrite=True)
        if True: #TODO #PRECOMMIT not translations["is_complete"]:
            descriptions = preprocess_raw_course_file(raw_descriptions)
            assert len(descriptions) == len(languages)
            to_translate = [i for i in zip(descriptions["Beschreibung"], languages.values(), languages.keys()) if i[1] != "en"]
            translateds, did_update, is_complete= translate_elems(*list(zip(*to_translate)), already_translated=translations["translations"])
            if did_update:
                json_persister.save(translations_file, translations=translateds, is_complete=is_complete, relevant_params=[], force_overwrite=True)
            translations = json_persister.load(translations_file, "translated_descriptions", ignore_params=["translate_policy"], force_overwrite=True)
            if not is_complete:
                print("The translated Descriptions are not complete yet!")
                exit(1)
            else:
                print("The translated Descriptions are now complete! Yay!")

        return translations, languages
    return None, None


########################################################################################################################
################################################ OLD STUFF #############################################################
########################################################################################################################


#TODO: create languages-file from titles & descriptions, see
# create_load_languages_file(from_path, names, descriptions)
# derive_conceptualspace.create_spaces.translate_descriptions.create_load_languages_file


# def create_load_languages_file(file_path, names, descriptions, filename="languages.json"):
#     if not isfile(join(file_path, filename)):
#         lans = []
#         print("Finding languages of descriptions...")
#         for desc in tqdm(descriptions):
#             try:
#                 lans.append(detect(desc))
#             except LangDetectException as e:
#                 lans.append("unk")
#         lans = dict(zip(names, lans))
#         with open(join(file_path, filename), "w") as ofile:
#             json.dump(lans, ofile)
#     else:
#         with open(join(file_path, filename), "r") as rfile:
#             lans = json.load(rfile)
#         if set(names)-set(lans):
#             missing_inds = [n for n, name in enumerate(names) if name not in lans]
#             print("Finding languages of descriptions...")
#             for ind, desc in tqdm([[i, list(descriptions)[i]] for i in missing_inds]):
#                 try:
#                     lans[list(names)[ind]] = detect(desc)
#                 except LangDetectException as e:
#                     lans[list(names)[ind]] = "unk"
#             with open(join(file_path, filename), "w") as ofile:
#                 json.dump(lans, ofile)
#     return lans
#
#
# def translate_descriptions(base_dir, mds_basename):
#     ndm_file = next(i for i in os.listdir(base_dir) if i.startswith(mds_basename) and i.endswith(".json"))
#     mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=ORIGLAN)
#     names, descriptions, mds, languages = mds_obj.names, mds_obj.descriptions, mds_obj.mds, mds_obj.languages
#     #TODO use languages
#     assert len(set(names)) == len(names)
#     descriptions = [html.unescape(i) for i in descriptions]
#     name_desc = dict(zip(names, descriptions))
#     if isfile((translationsfile := join(base_dir, "translated_descriptions.json"))):
#         with open(translationsfile, "r") as rfile:
#             translateds = json.load(rfile)
#     else:
#         translateds = {}
#     languages = create_load_languages_file(names, descriptions)
#     untranslated = {k:v for k,v in name_desc.items() if languages[k] != "en" and k not in translateds}
#     if not untranslated:
#         print("Everything is translated!!")
#         return
#     print(f"There are {len(''.join([i[0] for i in untranslated.values()]))} descriptions to be translated.")
#     translations = translate_text([name_desc[k] for k in untranslated], origlans=[languages[k] for k in untranslated])
#     # hash_translates = dict(zip([hashlib.sha256(i.encode("UTF-8")).hexdigest() for i in to_translate], translations))
#     translateds.update(dict(zip(untranslated.keys(), translations)))
#     with open(join(base_dir, "translated_descriptions.json"), "w") as wfile:
#         json.dump(translateds, wfile)
#     translation_new_len = len("".join(translations))
#     translated_descs = [name_desc[i] for i in name_desc.keys() if i in set(dict(zip([k for k in untranslated], translations)).keys())]
#     print(f"You translated {len('.'.join(translated_descs))} (becoming {translation_new_len}) Chars from {len(translated_descs)} descriptions.")
#
#
# def count_translations(base_dir, mds_basename=None, descriptions_basename=None):
#     assert not (mds_basename and descriptions_basename)
#     if mds_basename:
#         raise NotImplementedError("Using `load_translate_mds` would be a circular import!")
#         # ndm_file = next(i for i in os.listdir(base_dir) if i.startswith(mds_basename) and i.endswith(".json"))
#         # mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=ORIGLAN)
#         # names, descriptions, embedding, languages = mds_obj.names, mds_obj.descriptions, mds_obj.embedding, mds_obj.languages
#     else:
#         descriptions = [Description.fromstruct(i[1]) for i in json_load(join(base_dir, descriptions_basename))["descriptions"]]
#         names = [desc.title for desc in descriptions]
#         descriptions  = [desc.orig_text for desc in descriptions]
#     assert len(set(names)) == len(names)
#     name_desc = dict(zip(names, descriptions))
#     if isfile((translationsfile := join(base_dir, "translated_descriptions.json"))):
#         with open(translationsfile, "r") as rfile:
#             translateds = json.load(rfile)
#     else:
#         translateds = {}
#     languages = create_load_languages_file(base_dir, names, descriptions)
#     all_untranslateds = {k: v for k, v in name_desc.items() if languages[k] != "en" and k not in translateds}
#     all_translateds = {k: v for k, v in name_desc.items() if languages[k] != "en" and k in translateds}
#     all_notranslateds = {k: v for k, v in name_desc.items() if languages[k] == "en"}
#
#     print("Regarding #Descriptions:")
#     print(f"{len(all_untranslateds)+len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)} ({round((len(all_untranslateds)+len(all_translateds))/(len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)), 4)*100}%) need to be translated")
#     print(f"{len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)} ({round((len(all_translateds))/(len(all_untranslateds)+len(all_translateds)), 4)*100}%) are translated")
#     print("Reagarding #Chars:")
#     all_untranslateds, all_translateds, all_notranslateds = "".join(list(all_untranslateds.values())), "".join(list(all_translateds.values())), "".join(list(all_notranslateds.values()))
#     print(f"{len(all_untranslateds)+len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)} ({round((len(all_untranslateds)+len(all_translateds))/(len(all_untranslateds)+len(all_translateds)+len(all_notranslateds)), 4)*100}%) need to be translated")
#     print(f"{len(all_translateds)}/{len(all_untranslateds)+len(all_translateds)} ({round((len(all_translateds))/(len(all_untranslateds)+len(all_translateds)), 4)*100}%) are translated")
