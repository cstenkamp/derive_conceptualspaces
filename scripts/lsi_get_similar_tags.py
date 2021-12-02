from os.path import join, isfile, dirname, basename
import logging
import argparse

import pandas as pd
import numpy as np

from src.main.util.pretty_print import pretty_print as print
from scripts.create_siddata_dataset import ORIGLAN, ONLYENG, TRANSL, load_translate_mds  # TODO why is this in scripts
from src.main.load_data.siddata_data_prep.jsonloadstore import json_load
from src.main.util.text_tools import phrase_in_text, tokenize_text

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


def main():
    data_base_dir = parse_command_line_args().path
    mds_obj = load_translate_mds(data_base_dir, f"siddata_names_descriptions_mds_3.json", translate_policy=TRANSL)
    candidate_terms = json_load(join(data_base_dir, "candidate_terms.json"))
    assert isfile(join(data_base_dir, "candidate_terms_existinds.json"))
    mds_obj.term_existinds = json_load(join(data_base_dir, "candidate_terms_existinds.json"), assert_meta=("MDS_DIMENSIONS", "CANDIDATETERM_MIN_OCCURSIN_DOCS", "STANFORDNLP_VERSION"))
    print("The 25 candidate_terms that occur in the most descriptions (incl the #descriptions they occur in):", {i[0]: len(i[1]) for i in sorted(mds_obj.term_existinds.items(), key=lambda x: len(x[1]), reverse=True)[:25]})
    # assert set(flatten(candidate_terms)) == set(mds_obj.term_existinds.keys()) #TODO why is this not the case?
    TDM_COUNT = False #im Paper machen die `tag-applied`, which is boolean
    term_doc_matrix = pd.DataFrame(np.zeros([len(mds_obj.names), len(set(mds_obj.term_existinds.keys()))]), mds_obj.names, list(set(mds_obj.term_existinds.keys())))
    err_count = 0
    for term, indices in mds_obj.term_existinds.items():
        print(term)
        for ind in indices:
            # print(term_doc_matrix.loc[:, term].index[ind])
            # print(mds_obj.descriptions[ind])
            res = phrase_in_text(term, mds_obj.descriptions[ind], return_count=TDM_COUNT)
            if res:
                term_doc_matrix.loc[:, term].iloc[ind] += (1 if not TDM_COUNT else res)
                #TODO er sollte sie ja eigentlich normieren, oder TF/IDF, oder oder oder sonst macht das doch keinen sinn..
            else:
                err_count += 1
                #TODO Why the fuck?!
        print()
    if TDM_COUNT:
        print(f"Max-Count: Doc {term_doc_matrix.index[np.unravel_index(term_doc_matrix.to_numpy().argmax(), term_doc_matrix.to_numpy().shape)[0]]} with Term {term_doc_matrix.columns[np.unravel_index(term_doc_matrix.to_numpy().argmax(), term_doc_matrix.to_numpy().shape)[1]]}")
    pseudo_tdm = pd.DataFrame(np.eye(len(set(mds_obj.term_existinds.keys()))), list(set(mds_obj.term_existinds.keys())), list(set(mds_obj.term_existinds.keys())))


if __name__ == '__main__':
    main()
