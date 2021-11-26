from collections import Counter
from os.path import join, isfile, dirname, basename
import logging
import json
from datetime import datetime
import argparse


import sklearn.svm
import numpy as np
from tqdm import tqdm

from main.util.telegram_notifier import telegram_notify
from src.static.settings import SID_DATA_BASE, CANDIDATETERM_MIN_OCCURSIN_DOCS
from src.main.util.pretty_print import pretty_print as print
from scripts.create_siddata_dataset import ORIGLAN, ONLYENG, TRANSL, load_translate_mds #TODO why is this in scripts
from src.main.create_spaces.text_tools import phrase_in_text, tokenize_text
from src.main.load_data.siddata_data_prep.jsonloadstore import json_dump, json_load

from src.main.util.threedfigure import ThreeDFigure, make_meshgrid
from src.main.util.base_changer import Plane, make_base_changer

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()

def main():
    args = parse_command_line_args()
    data_base_dir = args.path
    cache_file = join(data_base_dir, "candidate_terms_existinds.json")
    mds_obj = load_translate_mds(data_base_dir, f"siddata_names_descriptions_mds_3.json", translate_policy=TRANSL)

    if isfile(cache_file):
        print(f"Loading the exist-indices-file from cache at {cache_file}!")
        mds_obj.term_existinds = json_load(cache_file, assert_meta=("MDS_DIMENSIONS", "CANDIDATETERM_MIN_OCCURSIN_DOCS", "STANFORDNLP_VERSION"))
    else:
        candidate_terms = json_load(join(data_base_dir, "candidate_terms.json"))
        if "candidate_terms" in candidate_terms: candidate_terms = candidate_terms["candidate_terms"]
        # if len(candidate_terms) < len(names):
        #     print(f"There are {len(names)} descriptions, but only candidate_terms for {len(candidate_terms)}!")
        #     names = names[:len(candidate_terms)]
        #     descriptions = descriptions[:len(candidate_terms)]  # TODO remove this this is baad
        #     mds.embedding_ = mds.embedding_[:len(descriptions):]
        assert len(candidate_terms) == len(mds_obj.descriptions)
        # assert all(j in descriptions[i].lower() for i in range(len(descriptions)) for j in candidate_terms[i])
        for desc_ind, desc in enumerate(mds_obj.descriptions):
            for cand in candidate_terms[desc_ind]:
                assert cand in desc.lower()
                assert phrase_in_text(cand, desc)

        all_terms = list(set(flatten(candidate_terms)))
        print("Checking for all phrases and all descriptions if the phrase occurs in the description, this takes ~20mins for ~25k phrases and ~5.3k descriptions")
        exist_indices = {term: [ind for ind, cont in enumerate(descriptions) if phrase_in_text(term, cont)] for term in tqdm(all_terms)}
        term_existinds = {term:exist_indices for term, exist_indices in exist_indices.items() if len(exist_indices) >= CANDIDATETERM_MIN_OCCURSIN_DOCS}
        term_existinds = dict(sorted(term_existinds.items(), key=lambda x: len(x[1]), reverse=True))
        json_dump(term_existinds, cache_file)
        print(f"Found {len(exist_indices)} candidate Terms, {len(term_existinds)} of which occur in at least {CANDIDATETERM_MIN_OCCURSIN_DOCS} descriptions.")
        print("The 25 terms that are most often detected as candidate terms (incl. their #dectections):",
            sorted(dict(Counter(flatten(candidate_terms))).items(), key=lambda x: x[1], reverse=True)[:25])

    print("The 25 candidate_terms that occur in the most descriptions (incl the #descriptions they occur in):",
          {i[0]: len(i[1]) for i in sorted(mds_obj.term_existinds.items(), key=lambda x: len(x[1]), reverse=True)[:25]})

    correct_percentages = {}
    for term, exist_indices in tqdm(mds_obj.term_existinds.items()):
        labels = [False] * len(mds_obj.names)
        for i in exist_indices:
            labels[i] = True
        # TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
        svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
        svm.fit(mds_obj.mds.embedding_, np.array(labels, dtype=int))
        svm_results = svm.decision_function(mds_obj.mds.embedding_)
        correct_preds = [labels[i] == (svm_results[i] > 0) for i in range(len(labels))]
        correct_percentage = round(sum(correct_preds)/len(correct_preds), 4)*100
        correct_percentages[term] = correct_percentage
        # print(f"Correct Percentage: {correct_percentage}%")

        # decision_plane = Plane(*svm.coef_[0], svm.intercept_[0])
        # with ThreeDFigure() as fig:
        #     X = mds_obj.mds.embedding_
        #     y = np.array(labels, dtype=int)
        #     fig.add_markers(X, color=y, size=1)  # samples
        #
        #     trafo, back_trafo = make_base_changer(decision_plane)
        #     onto_plane = np.array([back_trafo([0, trafo(point)[1], trafo(point)[2]]) for point, side in zip(X, y)])
        #     minx, miny = onto_plane.min(axis=0)[:2]
        #     maxx, maxy = onto_plane.max(axis=0)[:2]
        #     xx, yy = make_meshgrid(minx=minx, miny=miny, maxx=maxx, maxy=maxy, margin=0.1)
        #
        #     fig.add_surface(xx, yy, decision_plane.z)  # decision hyperplane
        #     fig.add_line(X.mean(axis=0) - decision_plane.normal*50, X.mean(axis=0) + decision_plane.normal*10, width=50)  # orthogonal of decision hyperplane through mean of points
        #     fig.add_markers([0, 0, 0], size=10)  # coordinate center
        #     # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        #     # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        #     fig.show()
        # print()
    print(f"Average Correct Percentages: {round(sum(list(correct_percentages.values()))/len(list(correct_percentages.values())), 2)}%")
    sorted_percentages = sorted([[k,round(v,2)] for k,v in correct_percentages.items()], key=lambda x:x[1], reverse=True)
    best_ones = list(dict(sorted_percentages).keys())[:50]
    best_dict = {i: [f"{round(correct_percentages[i], 2)}%", f"{len(mds_obj.term_existinds[i])} samples"] for i in best_ones}
    for k, v in best_dict.items():
        print(f"{k}: {'; '.join(v)}")




if __name__ == '__main__':
    main()
