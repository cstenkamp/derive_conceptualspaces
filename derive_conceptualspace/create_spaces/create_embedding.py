import logging
import math
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
    assert dissim_measure in ["cosine", "norm_ang_dist", "euclidian"]
    if isinstance(arr, scipy.sparse.csr.csr_matrix):
        arr = arr.toarray().T
    assert arr.shape[0] < arr.shape[1], "I cannot believe your Doc-Term-Matrix has less distinct words then documents."
    assert arr.max(axis=1).min() > 0, "If one of the vectors is zero the calculation will fail!"
    # return squareform(np.apply_along_axis(cos_to_normangdiff, 0, pdist(arr, metric="cosine")))
    # assert np.allclose(np.hstack([cdist(arr, arr[i*10:(i+1)*10], "cosine") for i in range(10)]), squareform(tmp))
    if dissim_measure in ["cosine", "norm_ang_dist"]:
        dist_func = "cosine"
    else:
        dist_func = dissim_measure
    tmp = []
    N_CHUNKS = 200
    n_procs = get_ncpu(ram_per_core=10)   # max. 1 thread per 10GB RAM
    if n_procs > 1:
        print(f"Running with {n_procs} Processes")
        with WorkerPool(n_procs, arr, pgbar="Creating dissimilarity matrix") as pool:
            tmp = pool.work(list(np.array_split(arr, N_CHUNKS)), lambda arr, chunk: cdist(arr, chunk, dist_func))
    else:
        for chunk in tqdm(np.array_split(arr, N_CHUNKS), desc="Creating dissimilarity matrix"):
            tmp.append(cdist(arr, chunk, dist_func))
    assert np.allclose(np.hstack(tmp), np.hstack(tmp).T), "The matrix must be symmetric!"
    res = np.hstack(tmp)
    if dissim_measure == "norm_ang_dist":
        flat = squareform(np.hstack(tmp), checks=False) #dunno why this one fails though np.hstack(tmp) == np.hstack(tmp).T
        res = squareform(np.apply_along_axis(cos_to_normangdiff, 0, flat))
    return res


def create_embedding(dissim_mat, embed_dimensions, embed_algo, verbose=False, pp_descriptions=None):
    dissim_mat = (dissim_mat[0], (1 - dissim_mat[1])-np.eye(dissim_mat[1].shape[0])) #motherfucker I calculated similarity and need Dissimilarity
    if embed_algo == "mds":
        embed = create_mds(dissim_mat, embed_dimensions)
    elif embed_algo == "tsne":
        embed = create_tsne(dissim_mat, embed_dimensions)
    elif embed_algo == "isomap":
        embed = create_isomap(dissim_mat, embed_dimensions)
    else:
        raise NotImplementedError(f"Algorithm {embed_algo} is not implemented!")
    if verbose and pp_descriptions is not None:
        from scipy.spatial.distance import squareform, pdist
        new_dissim = squareform(pdist(embed.embedding_))
        show_close_descriptions(new_dissim, pp_descriptions, num=10, title="Embedding")
    return embed


def create_mds(dissim_mat, embed_dimensions):
    dtm, dissim_mat = dissim_mat
    dissim_mat = dissim_mat*1000
    #TODO - isn't isomap better suited than MDS? https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling
    # !! [DESC15] say they compared it and it's worse ([15] of [DESC15])!!!
    n_inits = math.ceil((max(get_ncpu()*2, 10))/get_ncpu())*get_ncpu() # minimally 10, maximally ncpu*2, but in any case a multiple of ncpu
    embedding = MDS(n_components=embed_dimensions, dissimilarity="precomputed",
                    metric=False, #TODO with metric=True it always breaks after the second step if  n_components>>2
                    n_jobs=get_ncpu(), verbose=1 if get_setting("VERBOSE") else 0, n_init=n_inits, max_iter=3000, eps=1e-7)
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





def show_close_descriptions(dissim_mat, descriptions, num=10, title="Dissim-Mat"):
    # closest_entries = list(zip(*np.where(dissim_mat==min(dissim_mat[dissim_mat>0]))))
    # closest_entries = set(tuple(sorted(i)) for i in closest_entries)
    # print(f"Closest Nonequal Descriptions: \n", "\n".join(["*b*"+("*b* & *b*".join([descriptions._descriptions[i].title for i in j]))+"*b*" for j in closest_entries]))
    min_vals = sorted(squareform(dissim_mat))[:num]
    min_indices = np.where(np.isin(dissim_mat, min_vals))
    min_indices = [(i,j) for i,j in zip(*min_indices) if i!=j]
    min_indices = list({j: None for j in [tuple(sorted(i)) for i in min_indices]}.keys()) #remove duplicates ("aircraft cabin and airplane cabin" and "airplane cabin and aircraft cabin")
    print(f"Closest {num} Descriptions in {title}:")
    for first, second in min_indices[:num]:
        print(f"  *b*{descriptions._descriptions[first].title}*b* and *b*{descriptions._descriptions[second].title}*b*")
