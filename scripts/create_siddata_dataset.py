"""The purpose of this file is to create a dataset from the Siddata-data that looks like the three datasets used in [DESC15],
available at http://www.cs.cf.ac.uk/semanticspaces/. Meaning: MDS, ..."""
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import re
import random
import logging
import os

import numpy as np
import pandas as pd

from src.static.settings import SID_DATA_BASE, DEBUG, RANDOM_SEED, DATA_BASE, DATA_SET, MDS_DIMENSIONS
from src.main.util.logging import setup_logging
from src.main.util.pretty_print import pretty_print as print
from src.main.data_prep.create_mds import preprocess_data
from src.main.data_prep.jsonloadstore import json_dump, json_load

logger = logging.getLogger(basename(__file__))

##################################################################################################

def main():
    setup_logging("INFO")
    random.seed(RANDOM_SEED)
    for n_dims in [20, 100]: #[20,50,100,200]: #TODO #PRECOMMIT
        create_dataset(n_dims, "courses")


def create_dataset(n_dims, dsetname):
    # assert not DEBUG #TODO #PRECOMMIT
    names, descriptions, mds = create_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{n_dims}.json"), n_dims=n_dims)
        # names, descriptions, mds = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{n_dims}.json")) #TODO #PRECOMMIT comment out other line
    display_mds(mds, names)
    fname = join(DATA_BASE, dsetname, f"d{n_dims}", f"{dsetname}{n_dims}.mds")
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


def load_mds(data_path, assert_meta=()):
    loaded = json_load(data_path, assert_meta=assert_meta)
    names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]
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
    main()



