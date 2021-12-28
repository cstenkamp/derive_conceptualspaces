import logging
import math
from os.path import basename

import numpy as np
import scipy.sparse.csr
from scipy.spatial.distance import squareform, cdist
from sklearn.manifold import MDS, TSNE
from tqdm import tqdm

from derive_conceptualspace.settings import get_setting

logger = logging.getLogger(basename(__file__))


norm_ang_diff = lambda v1, v2: 2/math.pi * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
cos_to_normangdiff = lambda cosine: 2/math.pi*(np.arccos(-cosine+1))
# https://stackoverflow.com/questions/35758612/most-efficient-way-to-construct-similarity-matrix
# https://www.kaggle.com/cpmpml/ultra-fast-distance-matrix-computation/notebook
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html -> it's 2/math.pi*(np.arccos(-cosine+1)
# it WOULD be squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine"))), but this way we don't have a progressbar

def create_dissimilarity_matrix(arr, full=False):
    """returns the dissimilarity matrix, needed as input for the MDS. Input is the dataframe
    that contains all ppmi's of all entities (entities are rows, columns are terms, cells are then the
    ppmi(e,t) for all entity-term-combinations. Output is the normalized angular difference between
    all entities ei,ej --> an len(e)*len(e) matrix. This is needed as input for the MDS.
    See [DESC15] section 3.4."""
    #TODO why is it only 1 minute for 1000*1000 but >40 hours for 8000*8000?! it should be factor 64, not factor 2400
    if isinstance(arr, scipy.sparse.csr.csr_matrix):
        arr = arr.toarray().T
    assert arr.shape[0] < arr.shape[1], "I cannot believe your Doc-Term-Matrix has less distinct words then documents."
    assert arr.max(axis=1).min() > 0, "If one of the vectors is zero the calculation will fail!"
    logger.info("Creating the dissimilarity matrix...")
    # return squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine")))
    # assert np.allclose(np.hstack([cdist(arr, arr[i*10:(i+1)*10], "cosine") for i in range(10)]), squareform(tmp))
    tmp = []
    N_CHUNKS = 200
    for chunk in tqdm(np.array_split(arr, N_CHUNKS)):
        tmp.append(cdist(arr, chunk, "cosine"))
    assert np.allclose(np.hstack(tmp), np.hstack(tmp).T), "The matrix must be symmetric!"
    flat = squareform(np.hstack(tmp), checks=False) #I check in the line above, dunno why this one fails if the upper works..
    res = squareform(np.apply_along_axis(cos_to_normangdiff, 0, flat))
    return res
    # res = np.zeros((arr.shape[0],arr.shape[0]))
    # with tqdm(total=round(((arr.shape[0]*arr.shape[0])-arr.shape[0])*(1 if full else 0.5))) as pbar:
    #     for n1, e1 in enumerate(arr):
    #         for n2, e2 in enumerate(arr):
    #             if not full and n2 < n1:
    #                 continue
    #             if n1 != n2:
    #                 p1 = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    #                 if 0 < p1-1 < 1e-12:
    #                     p1 = 1 #aufgrund von rundungsfehlern kann es >1 sein
    #                 res[n1,n2] = 2 / math.pi * math.acos(p1)
    #                 pbar.update(1)
    # if not full:
    #     res[res.T > 0] = res.T[res.T > 0]
    # assert np.allclose(res, res.T, atol=1e-10)
    # return res


def create_embedding(dissim_mat, embed_dimensions, embed_algo):
    embed_algo = "".join([i for i in embed_algo.lower() if i.isalpha()])
    if embed_algo == "mds":
        return create_mds(dissim_mat, embed_dimensions)
    elif embed_algo == "tsne":
        return create_tsne(dissim_mat, embed_dimensions)
    else:
        raise NotImplementedError(f"Algorithm {embed_algo} is not implemented!")

def create_mds(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
    #TODO - isn't isomap better suited than MDS? https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling
    # !! [DESC15] say they compared it and it's worse ([15] of [DESC15])!!!
    embedding = MDS(n_components=embed_dimensions, random_state=get_setting("RANDOM_SEED", default_none=True), dissimilarity="precomputed")
    mds = embedding.fit(dissim_mat)
    return mds


def create_tsne(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
    embedding = TSNE(n_components=embed_dimensions, random_state=get_setting("RANDOM_SEED", default_none=True), metric="precomputed")
    tsne = embedding.fit(dissim_mat)
    return tsne
