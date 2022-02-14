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

def create_dissim_mat(descriptions: DescriptionList, quantification_measure, verbose=False, **interrupt_kwargs):
    #Options here: get_setting("NGRAMS_IN_EMBEDDING"), get_setting("DISSIM_MAT_ONLY_PARTNERED")
    if get_setting("DEBUG"):
        descriptions._descriptions = descriptions._descriptions[:get_setting("DEBUG_N_ITEMS")]
    dtm, metainf = descriptions.generate_DocTermMatrix(min_df=2 if get_setting("DISSIM_MAT_ONLY_PARTNERED") else 1,
                                                       max_ngram=get_setting("MAX_NGRAM") if get_setting("NGRAMS_IN_EMBEDDING") else None,
                                                       do_tfidf=quantification_measure if quantification_measure in ["tfidf", "tf"] else None)
    assert any(" " in i for i in dtm.all_terms.values()) == (get_setting("NGRAMS_IN_EMBEDDING") and get_setting("MAX_NGRAM") > 1)
    quantification = dtm.apply_quant(quantification_measure, descriptions=descriptions, verbose=verbose) if not metainf.get("sklearn_tfidf") else dtm
    # das ist jetzt \textbf{v}_e with all e's as rows
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    assert all(len(set((lst := [i[0] for i in dtm]))) == len(lst) for dtm in quantification.dtm)
    dissim_mat, metainf = create_dissimilarity_matrix(quantification.as_csr(), dissim_measure=get_setting("dissim_measure"), metainf=metainf, **interrupt_kwargs)
    if metainf.get("NEWLY_INTERRUPTED"):
        return quantification, dissim_mat, metainf
    assert np.allclose(dissim_mat, dissim_mat.T) #if so it's a correct dissimilarity-matrix and we can do squareform to compress
    metainf["is_dissim"] = np.allclose(np.diagonal(dissim_mat), 0, atol=1e-10)
    if verbose:
        show_close_descriptions(dissim_mat, descriptions)
    dissim_mat = squareform(dissim_mat, checks=True) #saves > 50% storage space!
    return quantification, dissim_mat, metainf

    #TODO: When I calculate PPMI here, relative to all documents and all possible terms, ist das relevant/unintended dass
    # die Grundgesamtheit (alle possible terms) ja anders ist als wenn ich das später nur auf den keyphrases mache? Kann ich das optional gleihcsetzen?

