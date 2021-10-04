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

logger = logging.getLogger(basename(__file__))

ORIGLAN = 1
ONLYENG = 2
TRANSL = 3

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
def extract_candidateterms_keybert():
    names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"), translate_policy=TRANSL)
    extractor = KeyBertExtractor(False, faster=True)
    candidateterms = []
    for desc in tqdm(descriptions):
        tmp = extractor(desc)
        keyberts = tmp[0]
        if (ct := extract_coursetype(desc)) and ct not in keyberts:
            keyberts += [ct]
        candidateterms.append(keyberts)
    with open(join(SID_DATA_BASE, "candidate_terms.json"), "w") as wfile:
        write_cont = {"git_hash": get_commithash(), "settings": get_settings(), "date": str(datetime.now()),
                      "model": extractor.model_name, "candidate_terms": [list(i) for i in candidateterms]}
        json.dump(write_cont, wfile)


@cli.command()
def create_all_datasets():
    # for n_dims in [20,50,100,200]:
    #     create_dataset(n_dims, "courses")
    create_dataset(2, "courses")


@cli.command()
def translate_descriptions():
    names, descriptions, mds, languages = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    #TODO use langauges
    assert len(set(names)) == len(names)
    descriptions = [html.unescape(i) for i in descriptions]
    name_desc = dict(zip(names, descriptions))
    if isfile((translationsfile := join(SID_DATA_BASE, "translated_descriptions.json"))):
        with open(translationsfile, "r") as rfile:
            translateds = json.load(rfile)
    else:
        translateds = {}
    unknown = {}
    print("Checking Language of descriptions...")
    for name, desc in tqdm(name_desc.items()):
        if name not in translateds:
            try:
                if (lan := detect(desc)) != "en":
                    unknown[name] = [desc, lan]
            except LangDetectException as e:
                unknown[name] = [desc, "unk"]
    print(f"There are {len(''.join([i[0] for i in unknown.values()]))} signs to be translated.")
    to_translate = [i for i in unknown.keys() if i not in translateds]
    translations = translate_text([unknown[i][0] for i in to_translate])
    # hash_translates = dict(zip([hashlib.sha256(i.encode("UTF-8")).hexdigest() for i in to_translate], translations))
    translateds.update(dict(zip(to_translate, translations)))
    with open(join(SID_DATA_BASE, "translated_descriptions.json"), "w") as wfile:
        json.dump(translateds, wfile)
    translation_new_len = len("".join(translations))
    translated_descs = [name_desc[i] for i in name_desc.keys() if i in set(dict(zip(to_translate, translations)).keys())]
    print(f"You translated {len('.'.join(translated_descs))} (becoming {translation_new_len}) Chars from {len(translated_descs)} descriptions.")

@cli.command()
def count_translations():
    names, descriptions, mds = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    assert len(set(names)) == len(names)
    descriptions = [html.unescape(i) for i in descriptions]
    name_desc = dict(zip(names, descriptions))
    if isfile((translationsfile := join(SID_DATA_BASE, "translated_descriptions.json"))):
        with open(translationsfile, "r") as rfile:
            translateds = json.load(rfile)
    else:
        translateds = {}
    unknown = {}
    translated_texts = [val for key, val in name_desc.items() if key in set(translateds.keys())]
    n_translateds = len("".join(translated_texts))
    for name, desc in tqdm(name_desc.items()):
        if name not in translateds:
            try:
                if (lan := detect(desc)) != "en":
                    unknown[name] = [desc, lan]
            except LangDetectException as e:
                unknown[name] = [desc, "unk"]
    n_untranslateds = len("".join([i[0] for i in unknown.values()]))
    percent = f"{round((n_translateds/(n_untranslateds+n_translateds))*100)}%"
    print(f"You have translated {n_translateds}/{n_translateds+n_untranslateds} ({percent})")


########################################################################################################################

def create_dataset(n_dims, dsetname):
    assert not DEBUG
    names, descriptions, mds = create_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{n_dims}.json"), n_dims=n_dims)
    # names, descriptions, mds = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{n_dims}.json")) #TODO #PRECOMMIT comment out other line
    display_mds(mds, names)
    fname = join(SPACES_DATA_BASE, dsetname, f"d{n_dims}", f"{dsetname}{n_dims}.mds")
    os.makedirs(dirname(fname), exist_ok=True)
    embedding = list(mds.embedding_)
    # indices = np.argsort(np.array(names))
    # names = [names[i] for i in indices]
    # descriptions = [descriptions[i] for i in indices]
    # embedding = np.array([embedding[i] for i in indices])
    if isfile(namesfile := join(dirname(fname), "..", "courseNames.txt")):
        with open(namesfile, "r") as rfile:
            assert [i.strip() for i in rfile.readlines()] == [i.strip() for i in names]
    else:
        with open(namesfile, "w") as wfile:
            wfile.writelines("\n".join(names))

    if isfile(fname):
        raise FileExistsError(f"{fname} already exists!")
    np.savetxt(fname, embedding, delimiter="\t")



def create_mds(data_path, n_dims):
    df = get_data()
    kwargs = {"max_elems": 100} if DEBUG else {}
    names, descriptions, mds = preprocess_data(df, n_dims=n_dims, **kwargs)
    json_dump({"names": names, "descriptions": descriptions, "mds": mds}, data_path)
    return names, descriptions, mds


def load_mds(data_path, assert_meta=(), translate_policy=ORIGLAN):
    loaded = json_load(data_path, assert_meta=assert_meta)
    languages_file = join(dirname(data_path), "languages.json")
    if not isfile(languages_file):
        lans = []
        for desc in tqdm(loaded['descriptions']):
            try:
                lans.append(detect(desc))
            except LangDetectException as e:
                lans.append("unk")
        with open(languages_file, "w") as ofile:
            json.dump(dict(zip(loaded["names"], lans)), ofile)
    else:
        with open(languages_file, "r") as rfile:
            lans = json.load(rfile)
    names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]
    languages = [lans[i] for i in names]
    if translate_policy == ORIGLAN:
        pass
    elif translate_policy == ONLYENG:
        indices = [ind for ind, elem in enumerate(languages) if elem == "en"]
        names, descriptions, languages = [names[i] for i in indices], [descriptions[i] for i in indices], [languages[i] for i in indices]
        mds.embedding_ = np.array([mds.embedding_[i] for i in indices])
        mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in indices])
    elif translate_policy == TRANSL:
        with open(join(SID_DATA_BASE, "translated_descriptions.json"), "r") as rfile:
            translations = json.load(rfile)
        new_descriptions, new_indices = [], []
        for ind, name in enumerate(names):
            if lans[name] == "en":
                new_descriptions.append(descriptions[ind])
                new_indices.append(ind)
            elif name in translations:
                new_descriptions.append(translations[name])
                new_indices.append(ind)
        descriptions = new_descriptions
        names, languages = [names[i] for i in new_indices], [languages[i] for i in new_indices]
        mds.embedding_ = np.array([mds.embedding_[i] for i in new_indices])
        mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in new_indices])
    descriptions = [html.unescape(i) for i in descriptions]
    return names, descriptions, mds, languages

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


def get_data(min_desc_len=10):
    #TODO in exploration I also played around with Levenhsthein-distance etc!
    df = pd.read_csv(join(SID_DATA_BASE, "kurse-beschreibungen.csv"))
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
    return df



if __name__ == '__main__':
    cli()



