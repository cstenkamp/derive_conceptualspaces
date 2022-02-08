from os.path import basename
import logging

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import tf_idf, ppmi
from .create_embedding import create_dissimilarity_matrix, show_close_descriptions
from ..util.desc_object import DescriptionList
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.create_spaces.preprocess_descriptions import get_countvec
from derive_conceptualspace.util.dtm_object import csr_to_list, DocTermMatrix

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
########################################################################################################################
# pipeline to the MDS representation from course descriptions

def create_dissim_mat(descriptions: DescriptionList, quantification_measure, verbose=False):
    #Options here: get_setting("NGRAMS_IN_EMBEDDING"), get_setting("DISSIM_MAT_ONLY_PARTNERED")
    if get_setting("DEBUG"):
        descriptions._descriptions = descriptions._descriptions[:get_setting("DEBUG_N_ITEMS")]
    if hasattr(descriptions, "recover_settings") and quantification_measure in ["tfidf", "tf"]:
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
        pipe = Pipeline([("count", get_countvec(**descriptions.recover_settings, min_df=2 if get_setting("DISSIM_MAT_ONLY_PARTNERED") else 1)),
                        ("tfidf", TfidfTransformer(use_idf=(quantification_measure=="tfidf")))]).fit(descriptions.unprocessed_texts)
        aslist, all_words = csr_to_list(pipe.transform(descriptions.unprocessed_texts), pipe["count"].vocabulary_)
        dtm = DocTermMatrix(dtm=aslist, all_terms=all_words, quant_name=quantification_measure)
    else:
        dtm = descriptions.generate_DocTermMatrix(min_df=2 if get_setting("DISSIM_MAT_ONLY_PARTNERED") else 1)
    assert any(" " in i for i in dtm.all_terms.values()) == get_setting("NGRAMS_IN_EMBEDDING")
    quantification = dtm.apply_quant(quantification_measure, descriptions=descriptions, verbose=verbose)
    # das ist jetzt \textbf{v}_e with all e's as rows
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    assert all(len(set((lst := [i[0] for i in dtm]))) == len(lst) for dtm in quantification.dtm)
    dissim_mat = create_dissimilarity_matrix(quantification.as_csr(), dissim_measure=get_setting("dissim_measure"))
    assert np.allclose(dissim_mat, dissim_mat.T) #if so it's a correct dissimilarity-matrix and we can do squareform to compress
    is_dissim = np.allclose(np.diagonal(dissim_mat), 0, atol=1e-10)
    if verbose:
        show_close_descriptions(dissim_mat, descriptions)
    dissim_mat = squareform(dissim_mat, checks=True) #saves > 50% storage space!
    return quantification, dissim_mat, {"ngrams_in_embedding": get_setting("NGRAMS_IN_EMBEDDING"), "is_dissim": is_dissim}


