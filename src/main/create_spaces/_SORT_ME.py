from os.path import join, isfile, dirname, basename
import re
import logging
import os

import numpy as np
import pandas as pd
from src.static.settings import SID_DATA_BASE, DEBUG, RANDOM_SEED, SPACES_DATA_BASE, DATA_BASE, MDS_DEFAULT_BASENAME
from src.main.util.pretty_print import pretty_print as print
from src.main.load_data.siddata_data_prep.create_mds import preprocess_data
from src.main.load_data.siddata_data_prep.jsonloadstore import json_dump, json_load
from src.main.create_spaces.main import load_translate_mds

from src.main.util.mds_object import TRANSL, ORIGLAN, ONLYENG

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
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