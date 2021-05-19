"""The purpose of this file is to create a dataset from the Siddata-data that looks like the three datasets used in [DESC15],
available at http://www.cs.cf.ac.uk/semanticspaces/. Meaning: MDS, ..."""
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import re
import random
import logging

import numpy as np
import pandas as pd

from src.static.settings import SID_DATA_BASE
from static.settings import DEBUG, RANDOM_SEED
from src.main.util.logging import setup_logging
from src.main.util.pretty_print import pretty_print as print
from src.main.data_prep.create_mds import preprocess_data
from src.main.data_prep.jsonloadstore import json_dump, json_dumps, json_load

logger = logging.getLogger(basename(__file__))

def main():
    setup_logging("INFO")
    random.seed(RANDOM_SEED)
    # df = get_data()
    # kwargs = {"max_elems": 100} if DEBUG else {}
    # names, descriptions, mds = preprocess_data(df, **kwargs)
    # json_dump({"names": names, "descriptions": descriptions, "mds": mds}, "arg.json")
    loaded = json_load("arg.json")
    names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]

    mins = np.argmin(np.ma.masked_equal(mds.dissimilarity_matrix_, 0.0, copy=False), axis=0)
    for cmp1, cmp2 in enumerate(mins):
        print(f"*b*{names[cmp1]}*b* is most similar to *b*{names[cmp2]}*b*")
    print()


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



