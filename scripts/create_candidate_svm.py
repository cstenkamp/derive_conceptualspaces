from collections import Counter
from os.path import join, isfile, dirname, basename
import logging
import json

import sklearn.svm
import numpy as np

from src.static.settings import SID_DATA_BASE
from src.main.util.pretty_print import pretty_print as print
from scripts.create_siddata_dataset import ORIGLAN, ONLYENG, TRANSL, load_mds

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

def main():
    names, descriptions, mds, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"),
                                         translate_policy=TRANSL)
    with open(join(SID_DATA_BASE, "candidate_terms.json"), "r") as rfile:
        candidate_terms = json.load(rfile)
    names = names[:len(candidate_terms)]
    descriptions = descriptions[:len(candidate_terms)]  # TODO remove this this is baad
    mds.embedding_ = mds.embedding_[:len(descriptions):]
    assert all(j in descriptions[i].lower() for i in range(len(descriptions)) for j in candidate_terms[i])
    print("Top 25: ", sorted(dict(Counter(flatten(candidate_terms))).items(), key=lambda x: x[1], reverse=True)[:25])
    all_terms = set(flatten(candidate_terms))
    for term in all_terms:
        exist_indices = [ind for ind, cont in enumerate(descriptions) if term in cont.lower()]
        if len(exist_indices) > 5: #TODO what to do if smaller?
            svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
            #TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
            labels = [False] * len(names)
            for i in exist_indices:
                labels[i] = True
            svm.fit(mds.embedding_, np.array(labels, dtype=np.int))
            print()

if __name__ == "__main__":
    main()