from os.path import basename
import logging

from .create_embedding import create_dissimilarity_matrix

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.text_tools import tf_idf, ppmi
from derive_conceptualspace.util.dtm_object import DocTermMatrix

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
########################################################################################################################
# pipeline to the MDS representation from course descriptions

#TODO merge with the other apply-quant-function
def create_dissim_mat(pp_descriptions, quantification_measure):
    vocab, descriptions = pp_descriptions.values()
    # TODO: # vocab, counts = tokenize_sentences_nltk(descriptions) if use_nltk_tokenizer else tokenize_sentences_countvectorizer(descriptions) allow countvectorizer!
    dtm = DocTermMatrix(all_terms=vocab, descriptions=descriptions)
    if quantification_measure == "tf-idf":
        quantification = tf_idf(dtm, verbose=bool(get_setting("VERBOSE")), descriptions=descriptions)
    elif quantification_measure == "ppmi":
        quantification = ppmi(dtm, verbose=bool(get_setting("VERBOSE")), descriptions=descriptions)  # das ist jetzt \textbf{v}_e with all e's as rows
    elif quantification_measure == "count":
        quantification = dtm.dtm
    elif quantification_measure == "binary":
        quantification = [[[j[0],min(j[1],1)] for j in i] for i in dtm.dtm]
    else:
        raise NotImplementedError()
    quantification = DocTermMatrix({"doc_term_matrix": quantification, "all_terms": dtm.all_terms})
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    assert all(len(set((lst := [i[0] for i in dtm]))) == len(lst) for dtm in quantification.dtm)
    dissim_mat = create_dissimilarity_matrix(quantification.as_csr())
    return quantification, dissim_mat




