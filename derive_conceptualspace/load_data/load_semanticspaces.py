import os
from os.path import join, isdir, isfile, abspath, dirname, splitext
import json

import numpy as np


#TODO If I want to do that for classes as well, I have to create these as well, not only load
#TODO Do this ones also for the [ALBS20] and [AGKS18] datasets


def load_ppmi_weighted_feature_vectors(data_base, data_set, return_array=True):
    """
    Step 1 to obtain a vector-space representation of  documents is PPMI, where words that occur in the documents
    are weighted by how relevant for this one entity they are (frequent with entity e while being infrequent in the overall corpus).
    See [DESC15] Section 3.4, source: http://www.cs.cf.ac.uk/semanticspaces/ Step after this is MDS.
    :param data_base: base-dir for the data
    :param data_set: one of the base-datasets available in [DESC15] (movies, wines, places)
    :return: PPMI-Weighted feature Vectors for data_set dataset.
    """
    #caching this into a json is SO much faster
    if isfile(join(data_base, data_set, "tokens.json")):
        with open(join(data_base, data_set, "tokens.json"), "r") as rfile:
            return json.load(rfile)
    dir = join(data_base, data_set, "Tokens") if isdir(join(data_base, data_set, "Tokens")) else join(data_base, data_set, "Vectors")
    result = {}
    for file in os.listdir(dir):
        result[splitext(file)[0]] = {}
        with open(join(dir,file)) as rfile:
            for line in rfile.readlines()[1:]:
                word, freq = line.strip().split("\t")
                result[splitext(file)[0]][word] = freq
    with open(join(data_base, data_set, "tokens.json"), "w") as wfile:
        json.dump(result, wfile)
    return np.array(result) if return_array else result


def load_mds_representation(data_base, data_set, n_dims, return_array=True, fname_out=[]):
    """
    We cannot use the `load_ppmi_weighted_feature_vectors` directly, because they are too sparse and we need a geometric
    representation in which entities correspond to points and in which Euclidean distance is a meaningful measure of
    dissimilarity. So we use MultiDimensionalScaling to create an n-dimensional Euclidean space, in which each entity e_i
    is associated with a point p_i such that the Euclidean distance d(p_i,p_j) approximates the dissimilarity ang(e_i,e_j).
    Small n_dims: Representations mainly capture high-level properties of the entities -> better generalization of specific representations.
    Large n_dims: Representations preserve more specific details, at the cost of being more noisy
    MDS created using MDSJ java library. SVD could be a possible alternative.
    See [DESC15] Section 3.4, source: http://www.cs.cf.ac.uk/semanticspaces/
    :param data_base: base-dir for the data
    :param data_set: one of the base-datasets available in [DESC15] (movies, wines, places)
    :param n_dims: How many dimensions for the MDS result
    :return: MDS-Representations. Result is referred to as S_{place} or S_{movie} in [DESC15].
    """
    TRANSLATE_FNAME = {"movies": "films"}
    assert str(n_dims) in ["3", "20", "50", "100", "200"]
    fname = join(data_base, data_set, f"d{n_dims}", f"{TRANSLATE_FNAME.get(data_set, data_set)}{n_dims}.mds")
    fname_out.append(fname)
    res = []
    with open(fname, "r") as rfile:
        for line in rfile.readlines():
            l = [float(i) for i in line.strip().split("\t")]
            assert len(l) == n_dims
            res.append(l)
    return (np.array(res) if return_array else res), fname


def get_names(data_base, data_set):
    TRANSLATE_FNAME = {"movies": "filmNames.txt"}
    fname = join(data_base, data_set, TRANSLATE_FNAME.get(data_set, f"{data_set[:-1]}Names.txt"))
    with open(fname, "r") as rfile:
        names = [i.strip() for i in rfile.readlines()]
    if not len(set(names)) == len(names):
        print("!! The names-list is not unique after stripping! #TODO")
    return names, fname


def get_classes(data_base, data_set, **kwargs):
    if data_set == "movies":
        from src.main.load_data.dataset_specifics.movies import get_classes as get_movie_classes
        return get_movie_classes(data_base, **kwargs)
    if data_set == "courses":
        from src.main.load_data.dataset_specifics.courses import get_classes as get_courses_classes
        return get_courses_classes(data_base, **kwargs)
    raise NotImplementedError()


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    if data_set == "movies":
        from src.main.load_data.dataset_specifics.movies import get_candidateterms as get_movie_candidateterms
        return get_movie_candidateterms(data_base, data_set, n_dims, **kwargs)
    if data_set == "courses":
        # from src.main.load_data.dataset_specifics.courses import get_classes as get_courses_classes
        raise NotImplementedError() #TODO what are good candidate terms for courses??
    raise NotImplementedError()


def get_clusters(data_base, data_set, n_dims):
    clusters = {}
    fname = join(data_base, data_set, f"d{n_dims}", "clusters20.txt")
    with open(fname, "r") as rfile:
        txt = [i.strip() for i in rfile.readlines()]
    iterator = iter(txt)
    clustername = None
    for line in iterator:
        if line.startswith("Cluster"):
            ncluster = int(line[len("Cluster "):line.find(":")])
            assert ncluster == len(clusters)+1
            assert next(iterator) == ""
            clustername = line[line.find(":")+1:].strip()
            clusters[clustername] = set()
        else:
            clusters[clustername].add(line)
    return clusters


def get_grouped_candidates(data_base, data_set, mds_dimensions):
    canditerms = get_candidateterms(data_base, data_set, mds_dimensions)
    clusters = get_clusters(data_base, data_set, mds_dimensions)
    canditerm_clusters = []
    cluster_directions = {key: None for key in clusters.keys()}
    for words, poss, vec, origname in zip(*canditerms):
        for clustername, clustercont in clusters.items():
            if origname == clustername:
                canditerm_clusters.append(clustername)
                cluster_directions[clustername] = vec
                break
            if origname in clustercont:
                canditerm_clusters.append(clustername)
                break
        #get here if no cluster applies
        canditerm_clusters.append(None)
    canditerms = [i for i in zip(*canditerms, canditerm_clusters) if i[4] is not None]
    assert not any(i is None for i in cluster_directions.keys())
    return canditerms, cluster_directions