import logging
import math
import warnings
from os.path import basename
import os

import numpy as np
import scipy.sparse.csr
from scipy.spatial.distance import squareform, cdist
from sklearn.manifold import MDS, TSNE, Isomap
from tqdm import tqdm

from derive_conceptualspace.util.threadworker import WorkerPool
from derive_conceptualspace.settings import get_setting, get_ncpu
from misc_util.pretty_print import pretty_print as print
import psutil

logger = logging.getLogger(basename(__file__))


norm_ang_diff = lambda v1, v2: 2/math.pi * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
cos_to_normangdiff = lambda cosine: 2/math.pi*(np.arccos(-cosine+1))
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html -> it's 2/math.pi*(np.arccos(-cosine+1)

# https://stackoverflow.com/questions/35758612/most-efficient-way-to-construct-similarity-matrix
# https://www.kaggle.com/cpmpml/ultra-fast-distance-matrix-computation/notebook
# it WOULD be squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine"))), but this way we don't have a progressbar

def create_dissimilarity_matrix(arr, dissim_measure):
    """returns the dissimilarity matrix, needed as input for the MDS. Input is the dataframe
    that contains all ppmi's of all entities (entities are rows, columns are terms, cells are then the
    ppmi(e,t) for all entity-term-combinations. Output is the normalized angular difference between
    all entities ei,ej --> an len(e)*len(e) matrix. This is needed as input for the MDS.
    See [DESC15] section 3.4."""
    assert dissim_measure in ["cosine", "norm_ang_dist", "euclidian"]
    if isinstance(arr, scipy.sparse.csr.csr_matrix):
        arr = arr.toarray().T
    assert arr.shape[0] < arr.shape[1], "I cannot believe your Doc-Term-Matrix has less distinct words then documents."
    assert arr.max(axis=1).min() > 0, "If one of the vectors is zero the calculation will fail!"
    return _create_dissim_mat(arr, dissim_measure)

def _create_dissim_mat(arr, dissim_measure, force_singlethread=False, n_chunks=200, silent=False):
    # return squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine")))
    # assert np.allclose(np.hstack([cdist(arr, arr[i*10:(i+1)*10], "cosine") for i in range(10)]), squareform(tmp))
    if dissim_measure in ["cosine", "norm_ang_dist"]:
        dist_func = "cosine"
    else:
        dist_func = dissim_measure
    tmp = []
    if not force_singlethread and get_ncpu(ram_per_core=10) > 1: # max. 1 thread per 10GB RAM
        if not silent: print(f"Running with {get_ncpu(ram_per_core=10)} Processes")
        with WorkerPool(get_ncpu(ram_per_core=10), arr, pgbar="Creating dissimilarity matrix" if not silent else None) as pool:
            tmp = pool.work(list(np.array_split(arr, n_chunks)), lambda arr, chunk: cdist(arr, chunk, dist_func))
    else:
        iterable = np.array_split(arr, n_chunks) if silent else tqdm(np.array_split(arr, n_chunks), desc="Creating dissimilarity matrix")
        for chunk in iterable:
            tmp.append(cdist(arr, chunk, dist_func))
    assert np.allclose(np.hstack(tmp), np.hstack(tmp).T), "The matrix must be symmetric!"
    res = np.hstack(tmp)
    if dissim_measure == "norm_ang_dist":
        flat = squareform(np.hstack(tmp), checks=False) #dunno why this one fails though np.hstack(tmp) == np.hstack(tmp).T
        res = squareform(np.apply_along_axis(cos_to_normangdiff, 0, flat))
    assert np.allclose(np.diagonal(res), 0, atol=1e-10) or np.allclose(np.diagonal(res), 1, atol=1e-10), "Diagonal must be 1 or 0!"
    assert np.allclose(res, res.T), "The matrix must be symmetric!"
    return res


def create_embedding(dissim_mat, embed_dimensions, embed_algo, verbose=False, pp_descriptions=None):
    dtm, dissim_mat = dissim_mat
    if get_setting("DEBUG"):
        dissim_mat = dissim_mat[:get_setting("DEBUG_N_ITEMS"), :get_setting("DEBUG_N_ITEMS")]
    is_dissim = np.allclose(np.diagonal(dissim_mat), 0, atol=1e-10)
    if not is_dissim:
        print("Seems like you had a similarity matrix, not a dissimilarity matrix! Fixing it.")
        assert np.allclose(np.diagonal(dissim_mat), 1, atol=1e-10)
        assert dissim_mat.min() >= 0 and dissim_mat.max() <= 1
        dissim_mat = 1-dissim_mat
    if embed_algo == "mds":
        embed = create_mds(dissim_mat, embed_dimensions)
    elif embed_algo == "tsne":
        embed = create_tsne(dissim_mat, embed_dimensions)
    elif embed_algo == "isomap":
        embed = create_isomap(dissim_mat, embed_dimensions)
    else:
        raise NotImplementedError(f"Algorithm {embed_algo} is not implemented!")
    if verbose and pp_descriptions is not None:
        show_close_descriptions(embed.embedding_, pp_descriptions, is_embedding=True, num=10, title=f"Embedding-Distances ({get_setting('DISSIM_MEASURE')})")
    if hasattr(embed, "dissimilarity_matrix_") and np.allclose(embed.dissimilarity_matrix_, dissim_mat):
        print("Dropping the dissim-mat from the embedding - it only bloats and is the same as in the previous step.")
        embed.dissimilarity_matrix_ = None
    return embed


def create_mds(dissim_mat, embed_dimensions, metric=True, init_from_isomap=True):
    max_iter = 10000 if not get_setting("DEBUG") else 100
    if not init_from_isomap:
        warnings.warn("sklearn's MDS is broken!! Have to init from something, don't fucking ask why!")
        n_inits = math.ceil((max(get_ncpu()*2, (10 if not get_setting("DEBUG") else 3)))/get_ncpu())*get_ncpu() # minimally 10, maximally ncpu*2, but in any case a multiple of ncpu
        print(f"Running {'non-' if not metric else ''}metric MDS {n_inits} times with {get_ncpu(ignore_debug=True)} jobs for max {max_iter} iterations.")
        embedding = MDS(n_components=embed_dimensions, dissimilarity="precomputed",
                        metric=metric, #TODO with metric=True it always breaks after the second step if  n_components>>2 (well, mit metric=False auch^^)
                        n_jobs=get_ncpu(ignore_debug=True), verbose=1 if get_setting("VERBOSE") else 0, n_init=n_inits, max_iter=max_iter)
        mds = embedding.fit(dissim_mat)
    else:
        print(f"Running {'non-' if not metric else ''}metric MDS with {get_ncpu(ignore_debug=True)} jobs for max {max_iter} iterations, initialized from Isomap-Embeddings")
        embedding = MDS(n_components=embed_dimensions, dissimilarity="precomputed", metric=metric,
                        n_jobs=get_ncpu(ignore_debug=True), verbose=1 if get_setting("VERBOSE") else 0, n_init=1, max_iter=max_iter)
        try:
            isomap_init = create_isomap(dissim_mat, embed_dimensions, neighbor_factor=10).embedding_
        except ValueError: #There are significant negative eigenvalues...
            isomap_init = np.random.random((len(dissim_mat), embed_dimensions))*0.01
        mds = embedding.fit(dissim_mat, init=isomap_init)
    return mds


def create_tsne(dissim_mat, embed_dimensions):
    embedding = TSNE(n_components=embed_dimensions, random_state=get_setting("RANDOM_SEED"), metric="precomputed")
    tsne = embedding.fit(dissim_mat)
    return tsne


def create_isomap(dissim_mat, embed_dimensions, neighbor_factor=2, **kwargs):
    # https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling says isomap better suited than MDS, but DESC15 say they compared it and it's worse ([15] of [DESC15])!
    n_neighbors=min(max(5, dissim_mat.shape[0]//neighbor_factor), dissim_mat.shape[0]-1)
    print(f"Running Isomap with {get_ncpu(ignore_debug=True)} jobs for max {n_neighbors} neighbors.")
    embedding = Isomap(n_jobs=get_ncpu(ignore_debug=True), n_neighbors=n_neighbors, n_components=embed_dimensions, metric="precomputed", **kwargs)
    isomap = embedding.fit(dissim_mat)
    return isomap





def show_close_descriptions(dissim_mat, descriptions, is_embedding=False, num=10, title="Dissim-Mat"):
    # closest_entries = list(zip(*np.where(dissim_mat==min(dissim_mat[dissim_mat>0]))))
    # closest_entries = set(tuple(sorted(i)) for i in closest_entries)
    # print(f"Closest Nonequal Descriptions: \n", "\n".join(["*b*"+("*b* & *b*".join([descriptions._descriptions[i].title for i in j]))+"*b*" for j in closest_entries]))
    print(f"Closest {num} Descriptions in {title}:")
    if is_embedding:
        dissim_mat = _create_dissim_mat(dissim_mat, get_setting("DISSIM_MEASURE"), force_singlethread=len(dissim_mat)<500, silent=len(dissim_mat)<500)
    is_dissim = np.allclose(np.diagonal(dissim_mat), 0, atol=1e-10)
    assert is_dissim, "TODO now it's a similarity matrix"
    min_vals = sorted(squareform(dissim_mat))[:num]
    min_indices = np.where(np.isin(dissim_mat, min_vals))
    min_indices = [(i,j) for i,j in zip(*min_indices) if i!=j]
    min_indices = list({j: None for j in [tuple(sorted(i)) for i in min_indices]}.keys()) #remove duplicates ("aircraft cabin and airplane cabin" and "airplane cabin and aircraft cabin")
    for first, second in min_indices[:num]:
        print(f"  *b*{descriptions._descriptions[first].title}*b* and *b*{descriptions._descriptions[second].title}*b*")
