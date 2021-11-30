"""The purpose of this file is to create a dataset from the Siddata-data that looks like the three datasets used in [DESC15],
available at http://www.cs.cf.ac.uk/semanticspaces/. Meaning: MDS, ..."""

#TODO make (snakemake?) Pipeline that runs start to finish and creates the complete directory
import hashlib
from collections import Counter
from os.path import join, isfile, dirname, basename
import re
import random
import logging
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm
import html
import click
from sklearn.svm import SVC
from gensim import corpora

from src.main.create_spaces.postprocess_candidates import postprocess_candidates
from src.main.create_spaces.text_tools import tokenize_text, phrase_in_text
from src.main.util.telegram_notifier import telegram_notify
from src.static.settings import SID_DATA_BASE, DEBUG, RANDOM_SEED, SPACES_DATA_BASE, DATA_BASE
from src.main.util.logutils import setup_logging
from src.main.util.pretty_print import pretty_print as print
from src.main.load_data.siddata_data_prep.create_mds import preprocess_data
from src.main.load_data.siddata_data_prep.jsonloadstore import json_dump, json_load
from src.main.util.google_translate import translate_text
from src.main.create_spaces.get_candidates_stanfordnlp import get_continuous_chunks_a, get_continuous_chunks_b, \
    stanford_extract_nounphrases, download_activate_stanfordnlp
from src.main.create_spaces.get_candidates_keybert import KeyBertExtractor
from src.main.create_spaces.get_candidates_rules import extract_coursetype
from src.main.load_data.siddata_data_prep.jsonloadstore import get_commithash, get_settings
from src.main.util.mds_object import MDSObject
from src.main.create_spaces.text_tools import phrase_in_text, tokenize_text

from src.main.util.mds_object import TRANSL, ORIGLAN, ONLYENG

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

##################################################################################################
#cli main

@click.group()
@click.option(
    "--log",
    type=str,
    default="INFO",
    help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
)
@click.option(
    "--logfile",
    type=str,
    default="",
    help="logfile to log to. If not set, it will be logged to standard stdout/stderr",
)
def cli(log="INFO", logfile=None):
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    setup_logging(log, logfile)
    random.seed(RANDOM_SEED)


@cli.resultcallback()
def process_result(*args, **kwargs):
    """gets executed after the actual code. Prints time for bookkeeping"""
    print("Done at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

########################################################################################################################
#commands

@cli.command()
def extract_candidateterms_stanfordlp():
    names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    print(stanford_extract_nounphrases(nlp, descriptions[1]))



@cli.command()
@click.argument("base-dir", type=str)
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_doc_term_matrix(base_dir, json_filename="candidate_terms_postprocessed.json"):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith("siddata_names_descriptions_mds_") and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=TRANSL)
    candidate_terms = json_load(join(base_dir, json_filename))["candidate_terms"]
    assert len(candidate_terms) == len(mds_obj.descriptions)
    assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    all_terms = list(set(flatten(candidate_terms)))
    descriptions = [tokenize_text(i)[1] for i in mds_obj.descriptions]
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    dictionary = corpora.Dictionary([all_terms])
    doc_term_matrix = [sorted([(ind, phrase_in_text(elem, mds_obj.descriptions[j], return_count=True)) for ind,elem in enumerate(all_terms) if phrase_in_text(elem, mds_obj.descriptions[0])], key=lambda x:x[0]) for j in tqdm(range(len(mds_obj.descriptions)))]
    json_dump({"all_terms": all_terms, "doc_term_matrix": doc_term_matrix}, join(base_dir, "doc_term_matrix.json"))



@cli.command()
@click.argument("base-dir", type=str)
@click.argument("--postfix", type=str, default="")
def postprocess_candidateterms(base_dir, postfix=""):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith("siddata_names_descriptions_mds_") and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=TRANSL)
    candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    model = candidate_terms["model"]
    candidate_terms = candidate_terms["candidate_terms"]
    assert len(candidate_terms) == len(mds_obj.descriptions)
    candidate_terms = postprocess_candidates(candidate_terms, mds_obj.descriptions)
    assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    json_dump({"model": model, "candidate_terms": [list(i) for i in candidate_terms], "postprocessed": True}, join(base_dir, f"candidate_terms{postfix}.json"))
    print(f"Saved the post-processed model under candidate_terms{postfix}.json!")


@cli.command()
@click.argument("base-dir", type=str)
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert(base_dir):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith("siddata_names_descriptions_mds_") and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=TRANSL)
    extractor = KeyBertExtractor(False, faster=False)
    candidateterms = []
    n_immediateworking_ges, n_fixed_ges, n_errs_ges = 0, 0, 0
    for desc in tqdm(mds_obj.descriptions):
        keyberts, origextracts, (n_immediateworking, n_fixed, n_errs) = extractor(desc)
        if (ct := extract_coursetype(desc)) and ct not in keyberts:
            keyberts += [ct]
        candidateterms.append(keyberts)
        n_immediateworking_ges += n_immediateworking
        n_fixed_ges += n_fixed
        n_errs_ges += n_errs
    print(f"Immediately working: {n_immediateworking_ges}")
    print(f"Fixed: {n_fixed_ges}")
    print(f"Errors: {n_errs_ges}")
    json_dump({"model": extractor.model_name, "candidate_terms": [list(i) for i in candidateterms]}, join(base_dir, "candidate_terms.json"))

#
# @cli.command()
# def create_all_datasets():
#     # for n_dims in [20,50,100,200]:
#     #     create_dataset(n_dims, "courses")
#     create_descstyle_dataset(20, "courses")

@cli.command()
def translate_descriptions(base_dir):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith("siddata_names_descriptions_mds_") and i.endswith(".json"))
    mds_obj = load_translate_mds(base_dir, ndm_file, translate_policy=ORIGLAN)
    names, descriptions, mds, languages = mds_obj.names, mds_obj.descriptions, mds_obj.mds, mds_obj.languages
    #TODO use langauges
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

@cli.command()
def count_translations(base_dir):
    ndm_file = next(i for i in os.listdir(base_dir) if i.startswith("siddata_names_descriptions_mds_") and i.endswith(".json"))
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

########################################################################################################################
# pipeline to create desc15-style-dataset

def get_data(data_dir, fname, min_desc_len=10):
    """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
        dropping duplicates"""
    #TODO in exploration I also played around with Levenhsthein-distance etc!
    df = pd.read_csv(join(data_dir, fname))
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
    df["Name"] = df["Name"].str.strip()
    return df


def create_mds(to_data_name, n_dims, from_csv_path=SID_DATA_BASE, from_csv_name="kurse-beschreibungen.csv", to_data_path=SID_DATA_BASE):
    """Creates a JSON with the names, descriptions and MDS (in non-DESC15-format)"""
    df = get_data(from_csv_path, from_csv_name)
    kwargs = {"max_elems": 100} if DEBUG else {}
    names, descriptions, mds = preprocess_data(df, n_dims=int(n_dims), **kwargs)
    json_dump({"names": names, "descriptions": descriptions, "mds": mds}, join(to_data_path, to_data_name))
    return names, descriptions, mds


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


def load_translate_mds(file_path, file_name, translate_policy, assert_meta=(), translations_filename="translated_descriptions.json", assert_allexistent=True):
    print(f"Working with file {file_name}!")
    loaded = json_load(join(file_path, file_name), assert_meta=assert_meta)
    names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]
    if assert_allexistent:
        assert len(names) == len(descriptions) == mds.embedding_.shape[0]
    languages = create_load_languages_file(names, descriptions, file_path=file_path)
    orig_n_samples = len(names)
    additional_kwargs = {}
    if translate_policy == ORIGLAN:
        pass
    elif translate_policy == ONLYENG:
        indices = [ind for ind, elem in enumerate(languages) if elem == "en"]
        print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones")
        names, descriptions, languages = [names[i] for i in indices], [descriptions[i] for i in indices], [languages[i] for i in indices]
        mds.embedding_ = np.array([mds.embedding_[i] for i in indices])
        mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in indices])
    elif translate_policy == TRANSL:
        additional_kwargs["original_descriptions"] = descriptions
        with open(join(file_path, translations_filename), "r") as rfile:
            translations = json.load(rfile)
        new_descriptions, new_indices = [], []
        for ind, name in enumerate(names):
            if languages[name] == "en":
                new_descriptions.append(descriptions[ind])
                new_indices.append(ind)
            elif name in translations:
                new_descriptions.append(translations[name])
                new_indices.append(ind)
        dropped_indices = set(range(len(new_indices))) - set(new_indices)
        if dropped_indices:
            print(f"Dropped {len(names) - len(new_indices)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
        descriptions = new_descriptions
        names, languages = [names[i] for i in new_indices], [list(languages.values())[i] for i in new_indices]
        mds.embedding_ = np.array([mds.embedding_[i] for i in new_indices])
        mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in new_indices])
    descriptions = [html.unescape(i).replace("  ", " ") for i in descriptions]
    if assert_allexistent:
        assert len(names) == len(descriptions) == mds.embedding_.shape[0] == orig_n_samples
    return MDSObject(names, descriptions, mds, languages, translate_policy, orig_n_samples, **additional_kwargs)


def display_mds(mds, names, max_elems=30):
    """
    Args:
         mds: np.array or data_prep.jsonloadstore.Struct created from sklearn.manifold.MDS or sklearn.manifold.MSD
         name: list of names
         max_elems (int): how many to display
    """
    if hasattr(mds, "embedding_"):
        mds = mds.embedding_
    mins = np.argmin(np.ma.masked_equal(mds, 0.0, copy=False), axis=0)
    for cmp1, cmp2 in enumerate(mins):
        print(f"*b*{names[cmp1]}*b* is most similar to *b*{names[cmp2]}*b*")
        if max_elems and cmp1 >= max_elems-1:
            break


def create_descstyle_dataset(n_dims, dsetname, from_path=SID_DATA_BASE, from_name_base="siddata_names_descriptions_mds_{n_dims}.json", to_path=SPACES_DATA_BASE, translate_policy=ORIGLAN):
    names, descriptions, mds, languages = load_translate_mds(from_path, from_name_base.format(n_dims=n_dims), translate_policy)
    display_mds(mds, names)
    fname = join(to_path, dsetname, f"d{n_dims}", f"{dsetname}{n_dims}.mds")
    os.makedirs(dirname(fname), exist_ok=True)
    embedding = list(mds.embedding_)
    indices = np.argsort(np.array(names))
    names, descriptions, embedding = [names[i] for i in indices], [descriptions[i] for i in indices], np.array([embedding[i] for i in indices])
    if isfile(namesfile := join(dirname(fname), "..", "courseNames.txt")):
        with open(namesfile, "r") as rfile:
            assert [i.strip() for i in rfile.readlines()] == [i.strip() for i in names]
    else:
        with open(namesfile, "w") as wfile:
            wfile.writelines("\n".join(names))
    if isfile(fname):
        raise FileExistsError(f"{fname} already exists!")
    np.savetxt(fname, embedding, delimiter="\t")

########################################################################################################################

if __name__ == '__main__':
    cli()



