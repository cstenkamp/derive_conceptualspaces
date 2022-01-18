import logging
import math
from os.path import basename

import numpy as np
import scipy.sparse.csr
from scipy.spatial.distance import squareform, cdist
from sklearn.manifold import MDS, TSNE, Isomap
from tqdm import tqdm

from derive_conceptualspace.settings import get_setting
from misc_util.pretty_print import pretty_print as print

logger = logging.getLogger(basename(__file__))


norm_ang_diff = lambda v1, v2: 2/math.pi * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
cos_to_normangdiff = lambda cosine: 2/math.pi*(np.arccos(-cosine+1))
# https://stackoverflow.com/questions/35758612/most-efficient-way-to-construct-similarity-matrix
# https://www.kaggle.com/cpmpml/ultra-fast-distance-matrix-computation/notebook
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html -> it's 2/math.pi*(np.arccos(-cosine+1)
# it WOULD be squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine"))), but this way we don't have a progressbar

def create_dissimilarity_matrix(arr, dissim_measure):
    """returns the dissimilarity matrix, needed as input for the MDS. Input is the dataframe
    that contains all ppmi's of all entities (entities are rows, columns are terms, cells are then the
    ppmi(e,t) for all entity-term-combinations. Output is the normalized angular difference between
    all entities ei,ej --> an len(e)*len(e) matrix. This is needed as input for the MDS.
    See [DESC15] section 3.4."""
    assert dissim_measure in ["cosine", "norm_ang_dist"]
    if isinstance(arr, scipy.sparse.csr.csr_matrix):
        arr = arr.toarray().T
    assert arr.shape[0] < arr.shape[1], "I cannot believe your Doc-Term-Matrix has less distinct words then documents."
    assert arr.max(axis=1).min() > 0, "If one of the vectors is zero the calculation will fail!"
    # return squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine")))
    # assert np.allclose(np.hstack([cdist(arr, arr[i*10:(i+1)*10], "cosine") for i in range(10)]), squareform(tmp))
    if dissim_measure in ["cosine", "norm_ang_dist"]:
        dist_func = "cosine"
    tmp = []
    N_CHUNKS = 200
    for chunk in tqdm(np.array_split(arr, N_CHUNKS), desc="Creating dissimilarity matrix"):
        tmp.append(cdist(arr, chunk, dist_func))
    assert np.allclose(np.hstack(tmp), np.hstack(tmp).T), "The matrix must be symmetric!"
    res = np.hstack(tmp)
    if dissim_measure == "norm_ang_dist":
        flat = squareform(np.hstack(tmp), checks=False) #dunno why this one fails though np.hstack(tmp) == np.hstack(tmp).T
        res = squareform(np.apply_along_axis(cos_to_normangdiff, 0, flat))
    return res


def create_embedding(dissim_mat, embed_dimensions, embed_algo, verbose=False, pp_descriptions=None):
    if embed_algo == "mds":
        embed = create_mds(dissim_mat, embed_dimensions)
    elif embed_algo == "tsne":
        embed = create_tsne(dissim_mat, embed_dimensions)
    elif embed_algo == "isomap":
        embed = create_isomap(dissim_mat, embed_dimensions)
    else:
        raise NotImplementedError(f"Algorithm {embed_algo} is not implemented!")
    if verbose and pp_descriptions is not None:
        min_vals = sorted(squareform(embed.dissimilarity_matrix_))[:10]
        min_indices = np.where(np.isin(embed.dissimilarity_matrix_, min_vals))
        min_indices = [(i,j) for i,j in zip(*min_indices) if i!=j]
        print("Closest 10 Descriptions in Embedding:")
        for first, second in min_indices[:10]:
            print(f"  *b*{pp_descriptions._descriptions[first].title}*b* and *b*{pp_descriptions._descriptions[second].title}*b*")
    return embed


def create_mds(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
        #TODO - isn't isomap better suited than MDS? https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling
    # !! [DESC15] say they compared it and it's worse ([15] of [DESC15])!!!
    embedding = MDS(n_components=embed_dimensions, random_state=get_setting("RANDOM_SEED"), dissimilarity="precomputed")
    mds = embedding.fit(dissim_mat)
    return mds


def create_tsne(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
    embedding = TSNE(n_components=embed_dimensions, random_state=get_setting("RANDOM_SEED"), metric="precomputed")
    tsne = embedding.fit(dissim_mat)
    return tsne


def create_isomap(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
    embedding = Isomap(n_neighbors=min(5, dissim_mat.shape[0]-1), n_components=embed_dimensions, metric="precomputed")
    isomap = embedding.fit(dissim_mat)
    return isomap
