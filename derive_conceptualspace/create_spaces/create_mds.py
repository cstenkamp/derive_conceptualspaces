from functools import partial
import math
import logging
from os.path import basename

import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS
import scipy.sparse.csr


from derive_conceptualspace.settings import RANDOM_SEED
from derive_conceptualspace.util.np_tools import np_divide, np_log
from derive_conceptualspace.util.text_tools import print_quantification
from derive_conceptualspace.util.tokenizers import tokenize_sentences_countvectorizer
from derive_conceptualspace.util.jsonloadstore import Struct

logger = logging.getLogger(basename(__file__))



def pmi(doc_term_matrix, positive=False, verbose=False, mds_obj=None, descriptions=None):
    """
    calculation of ppmi/pmi ([DESC15] 3.4 first lines)
    see https://stackoverflow.com/a/58725695/5122790
    see https://www.overleaf.com/project/609bbdd6a07c203c38a07ab4
    """
    logger.info("Calculating PMIs...")
    #see doc_term_matrix.as_csr().toarray() - spalten pro doc und zeilen pro term
    words_per_doc = doc_term_matrix.as_csr().sum(axis=0)       #old name: col_totals
    total_words = words_per_doc.sum()                          #old name: total
    ges_occurs_per_term = doc_term_matrix.as_csr().sum(axis=1) #old name: row_totals
    expected = np.outer(ges_occurs_per_term, words_per_doc)
    expected = np_divide(expected, total_words)
    quantifications = np_divide(doc_term_matrix.as_csr(), expected)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        quantifications = np_log(quantifications)
    if positive:
        quantifications[quantifications < 0] = 0.0
    quantifications  = [[[i,elem] for i, elem in enumerate(quantifications[:,i]) if elem != 0] for i in range(quantifications.shape[1])]
    if verbose:
        print_quantification(doc_term_matrix, quantifications, mds_obj=mds_obj, descriptions=descriptions)
    return quantifications

ppmi = partial(pmi, positive=True)


def create_dissimilarity_matrix(arr, full=False):
    """returns the dissimilarity matrix, needed as input for the MDS. Input is the dataframe
    that contains all ppmi's of all entities (entities are rows, columns are terms, cells are then the
    ppmi(e,t) for all entity-term-combinations. Output is the normalized angular difference between
    all entities ei,ej --> an len(e)*len(e) matrix. This is needed as input for the MDS.
    See [DESC15] section 3.4."""
    if isinstance(arr, scipy.sparse.csr.csr_matrix):
        arr = arr.toarray().T
    assert arr.shape[0] < arr.shape[1], "I cannot believe your Doc-Term-Matrix has more distinct words then documents."
    assert arr.max(axis=1).min() > 0, "If one of the vectors is zero the calculation will fail!"
    logger.info("Creating the dissimilarity matrix...")
    res = np.zeros((arr.shape[0],arr.shape[0]))
    with tqdm(total=round(((arr.shape[0]*arr.shape[0])-arr.shape[0])*(1 if full else 0.5))) as pbar:
        for n1, e1 in enumerate(arr):
            for n2, e2 in enumerate(arr):
                if not full and n2 < n1:
                    continue
                if n1 != n2:
                    assert e1.max() and e2.max()
                    p1 = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                    if 0 < p1-1 < 1e-12:
                        p1 = 1 #aufgrund von rundungsfehlern kann es >1 sein
                    res[n1,n2] = 2 / math.pi * math.acos(p1)
                    pbar.update(1)
    if not full:
        res[res.T > 0] = res.T[res.T > 0]
    assert np.allclose(res, res.T, atol=1e-10)
    return res
