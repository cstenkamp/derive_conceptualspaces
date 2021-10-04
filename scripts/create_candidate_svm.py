from collections import Counter
from os.path import join, isfile, dirname, basename
import logging
import json
from datetime import datetime

import sklearn.svm
import numpy as np
from tqdm import tqdm

from src.static.settings import SID_DATA_BASE, CANDIDATETERM_MIN_OCCURSIN_DOCS
from src.main.util.pretty_print import pretty_print as print
from scripts.create_siddata_dataset import ORIGLAN, ONLYENG, TRANSL, load_mds
from main.create_spaces.text_tools import phrase_in_text

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

def main():
    names, descriptions, mds, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"), translate_policy=TRANSL)
    with open(join(SID_DATA_BASE, "candidate_terms.json"), "r") as rfile:
        candidate_terms = json.load(rfile)
        if hasattr(candidate_terms, "candidate_terms"): candidate_terms = candidate_terms["candidate_terms"]
    print("step 1", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    if len(candidate_terms) < len(names):
        print(f"There are {len(names)} descriptions, but only candidate_terms for {len(candidate_terms)}!")
        names = names[:len(candidate_terms)]
        descriptions = descriptions[:len(candidate_terms)]  # TODO remove this this is baad
        mds.embedding_ = mds.embedding_[:len(descriptions):]
    print("step 2", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    assert all(j in descriptions[i].lower() for i in range(len(descriptions)) for j in candidate_terms[i])
    all_terms = list(set(flatten(candidate_terms)))
    exist_indices = {term: [ind for ind, cont in enumerate(descriptions) if phrase_in_text(term, cont)] for term in tqdm(all_terms)}
    print("step 3", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    term_existinds = {term:exist_indices for term, exist_indices in exist_indices.items() if len(exist_indices) >= CANDIDATETERM_MIN_OCCURSIN_DOCS}
    print("step 4", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    print(f"Found {len(exist_indices)} candidate Terms, {len(term_existinds)} of which occur in at least {CANDIDATETERM_MIN_OCCURSIN_DOCS} descriptions.")
    print("The 25 terms that are most often detected as candidate terms (incl. their #dectections):",
          sorted(dict(Counter(flatten(candidate_terms))).items(), key=lambda x: x[1], reverse=True)[:25])
    print("The 25 candidate_terms that occur in the most descriptions (incl the #descriptions they occur in):",
          {i[0]: len(i[1]) for i in sorted(term_existinds.items(), key=lambda x: len(x[1]), reverse=True)[:25]})
    #
    # for term in all_terms:
    #
    #     if len(exist_indices) > CANDIDATETERM_MIN_OCCURSIN_DOCS: #TODO what to do if smaller?
    #         svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
    #         #TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
    #         labels = [False] * len(names)
    #         for i in exist_indices:
    #             labels[i] = True
    #         svm.fit(mds.embedding_, np.array(labels, dtype=np.int))

if __name__ == "__main__":
    main()