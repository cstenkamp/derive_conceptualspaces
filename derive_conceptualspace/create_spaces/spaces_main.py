from os.path import basename
import logging

import numpy as np
from scipy.spatial.distance import squareform

from .create_embedding import create_dissimilarity_matrix, show_close_descriptions

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import tf_idf, ppmi
from derive_conceptualspace.util.dtm_object import DocTermMatrix
from ..util.desc_object import DescriptionList
from misc_util.pretty_print import pretty_print as print

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
########################################################################################################################
# pipeline to the MDS representation from course descriptions

def create_dissim_mat(descriptions: DescriptionList, quantification_measure, verbose=False):
    #Options here: get_setting("NGRAMS_IN_EMBEDDING"), get_setting("DISSIM_MAT_ONLY_PARTNERED")
    if get_setting("DEBUG"):
        descriptions._descriptions = descriptions._descriptions[:100]
    dtm = descriptions.generate_DocTermMatrix(min_df=2 if get_setting("DISSIM_MAT_ONLY_PARTNERED") else 1)
    assert any(" " in i for i in dtm.all_terms.values()) == get_setting("NGRAMS_IN_EMBEDDING")
    quantification = dtm.apply_quant(quantification_measure, descriptions=descriptions, verbose=verbose)
    # das ist jetzt \textbf{v}_e with all e's as rows
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    assert all(len(set((lst := [i[0] for i in dtm]))) == len(lst) for dtm in quantification.dtm)
    dissim_mat = create_dissimilarity_matrix(quantification.as_csr(), dissim_measure=get_setting("dissim_measure"))
    if verbose:
        show_close_descriptions(dissim_mat, descriptions)
    return quantification, dissim_mat, {"ngrams_in_embedding": get_setting("NGRAMS_IN_EMBEDDING")}


