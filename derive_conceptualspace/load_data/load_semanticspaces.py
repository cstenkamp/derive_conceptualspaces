import os
from os.path import join, isdir, isfile, abspath, dirname, splitext
import json

import numpy as np

from derive_conceptualspace.util.base_changer import ThreeDPlane
from derive_conceptualspace.util.threedfigure import ThreeDFigure, make_meshgrid

flatten = lambda l: [item for sublist in l for item in sublist]

TRANSLATE_FNAME = {"movies": "films"}

#TODO If I want to do that for classes as well, I have to create these as well, not only load
#TODO Do this ones also for the [ALBS20] and [AGKS18] datasets
data_base, data_set, n_dims = "/home/chris/Documents/UNI_neu/Masterarbeit/data/semanticspaces/", "movies", 20


def main():
    canditerms, cluster_directions, mds_class_dict = get_all()
    three_dims = list(cluster_directions.keys())[:3]
    entities = {k: (v[1], np.array([v[2][k2] for k2 in three_dims])) for k, v in mds_class_dict.items()}
    display_svm(entities, {k: cluster_directions[k] for k in three_dims})


def display_svm(entities, cluster_directions):
    X = np.array([i[1] for i in entities.values()])
    assert X.shape[1] == 3
    extras = [{"Name": i[0], "Classes": i[1][0]} for i in list(entities.items())]
    with ThreeDFigure(name=",".join(cluster_directions.keys())) as fig:
        fig.add_markers(X, color="blue", custom_data=extras, name="samples")
        for num, (dirname, sw_ax, color) in enumerate(zip(cluster_directions.keys(), ["xz", "yz", None], ["red", "green", "yellow"])):
            # fig.add_surface(ThreeDPlane(np.eye(3)[num], 0), X, color="gray")
            fig.add_surface_old(*make_meshgrid(size=0.2), lambda _, __: 0, opacity=0.3, showlegend=True, color=color, name=dirname, swap_axes=sw_ax)
        fig.add_markers([0, 0, 0], size=3, name="Coordinate Center")  # coordinate center
        fig.show()


def get_all():
    feat_vecs = load_ppmi_weighted_feature_vectors(data_base, data_set)            #./Tokens/*  & Tokens.json
    # dict(len 38649): key: bag-of-words
    #TODO why is feat_vecs more than twice #elems than names?! Can I link BoW and names/MDS?!
    #TODO also, no term of the BoW contains a space, so I don't think I can match this with the keyphrases, wtf!
    all_terms = sorted(list(set(flatten([list(v.keys()) for v in feat_vecs.values()]))))
    mds = load_mds_representation(data_base, data_set, n_dims, return_array=True)  #./dXX/filmsXX.embedding
    # np.array 15000*n_dims
    names = get_names(data_base, data_set)                                         #./filmNames.txt
    #list 15000 long
    classes = get_classes(data_base, data_set)                                     #./classesXXXXX/*
    canditerms = get_candidateterms(data_base, data_set, n_dims)                   #./dXX/DirectionsHeal/*
    #4 lists of len 22903 (words, part-of-speech's, np.array of n_dims*1, word+pos)
    clusters = get_clusters(data_base, data_set, n_dims)                           #./dXX/clustersXX.txt
    canditerms, cluster_directions = get_grouped_candidates(data_base, data_set, n_dims, clusters=clusters, canditerms=canditerms)
    # merges get_candidateterms and get_clusters -> ./dXX/DirectionsHeal/* & ./dXX/clustersXX.txt
    #list of candidates that ARE in a cluster, and for each of those a list [words, parts-of-speech, vector, orig(words+POSs), clustername]. -- len 9429. And cluster_directions len 40
    proj1, proj2 = load_projections(data_base, data_set, n_dims)                   #./dXX/filmsXX.projected and ./dXX/projectionsXX.data
    assert all(any(np.allclose(i, proj) for i in cluster_directions.values()) for proj in proj1)
    #proj1: 40*20 (same as cluster_directions), proj2: 15000*40
    #proj2 are the distances to the origins of the respective dimensions (induced by the clusters), what induces the respective rankings! (see DESC15 p.24u)
    mds_dict = dict(zip(names, list(mds)))
    assert mds_dict.keys() == classes.keys()
    #soo let's use all data. For every movie, there's a something in `names`, `embedding`, `classes` and `proj2` => `mds_class_dict`
    #`canditerms` uses the `clusters`, `proj1` is the same as `cluster_directions`
    #missing: `feat_vecs` (idk how to map these to names), `all_terms` (extracted phrases are ngrams),
    mds_class_dict = dict(zip(mds_dict.keys(), list(zip(mds_dict.values(), classes.values(), proj2))))
    mds_class_dict = {k: (v[0], v[1], dict(zip(cluster_directions.keys(), v[2]))) for k,v in mds_class_dict.items()}
    return canditerms, cluster_directions, mds_class_dict
    #ok, in spirit of DESC15 I want a class that can find min and max and compare entities w.r.t. different dimensions


def load_projections(data_base, data_set, n_dims):
    return (
        np.loadtxt(join(data_base, data_set, f"d{n_dims}", "projections20.data")),
        np.loadtxt(join(data_base, data_set, f"d{n_dims}", f"{TRANSLATE_FNAME.get(data_set, data_set)}{n_dims}.projected"))
    )

def load_ppmi_weighted_feature_vectors(data_base, data_set):
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
    result = dict(sorted([(int(k),v) for k,v in result.items()], key=lambda x:x[0]))
    result = {k: {k2: int(v2) for k2,v2 in v.items()} for k,v in result.items()}
    with open(join(data_base, data_set, "tokens.json"), "w") as wfile:
        json.dump(result, wfile)
    return result


def load_mds_representation(data_base, data_set, n_dims, return_array=True, fname_out=None):
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
    fname_out = fname_out or []
    assert str(n_dims) in ["3", "20", "50", "100", "200"]
    fname = join(data_base, data_set, f"d{n_dims}", f"{TRANSLATE_FNAME.get(data_set, data_set)}{n_dims}.embedding")
    fname_out.append(fname)
    res = []
    with open(fname, "r") as rfile:
        for line in rfile.readlines():
            l = [float(i) for i in line.strip().split("\t")]
            assert len(l) == n_dims
            res.append(l)
    return (np.array(res) if return_array else res)#, fname


def get_names(data_base, data_set):
    TRANSLATE_FNAME = {"movies": "filmNames.txt"}
    fname = join(data_base, data_set, TRANSLATE_FNAME.get(data_set, f"{data_set[:-1]}Names.txt"))
    with open(fname, "r") as rfile:
        names = [i.strip() for i in rfile.readlines()]
    if not len(set(names)) == len(names):
        print("!! The names-list is not unique after stripping! #TODO")
    return names#, fname


def get_classes(data_base, data_set, **kwargs):
    if data_set == "movies":
        from derive_conceptualspace.load_data.dataset_specifics.movies import get_classes as get_movie_classes
        if "what" not in kwargs:
            print("Will use `Genres` as type. Alternatives are: Genres, Keywords, Ratings.")
            kwargs["what"] = "Genres"
        return get_movie_classes(data_base, **kwargs)
    if data_set == "courses":
        from derive_conceptualspace.load_data.dataset_specifics.courses import get_classes as get_courses_classes
        return get_courses_classes(data_base, **kwargs)
    raise NotImplementedError()


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    if data_set == "movies":
        from derive_conceptualspace.load_data.dataset_specifics.movies import get_candidateterms as get_movie_candidateterms
        return get_movie_candidateterms(data_base, data_set, n_dims, **kwargs)
    if data_set == "courses":
        # from src.main.load_data.dataset_specifics.courses import get_classes as get_courses_classes
        raise NotImplementedError() #TODO what are good candidate terms for courses??
    raise NotImplementedError()


def get_clusters(data_base, data_set, n_dims):
    clusters = {}
    fname = join(data_base, data_set, f"d{n_dims}", f"clusters{n_dims}.txt")
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
        elif not line.startswith("**"):
            clusters[clustername].add(line)
    return clusters


def get_grouped_candidates(data_base, data_set, embed_dimensions, clusters=None, canditerms=None):
    """
    returns a list of all those candidates that ARE in a cluster, and for each of those a list [words, parts-of-speech, vector, orig(words+POSs), clustername].
    And a list of all cluster-vectors
    """
    def check_cluster(clusters, origname, cluster_directions):
        for clustername, clustercont in clusters.items():
            if origname == clustername:
                cluster_directions[clustername] = vec
                return origname
            if origname in clustercont:
                return clustername
    canditerms = canditerms or get_candidateterms(data_base, data_set, embed_dimensions)
    clusters = clusters or get_clusters(data_base, data_set, embed_dimensions)
    alls = set(flatten(list(clusters.values()))) | set(clusters.keys())
    assert not alls-set(canditerms[3])
    canditerm_clusters = []
    cluster_directions = {key: None for key in clusters.keys()}
    for words, poss, vec, origname in zip(*canditerms):
        if check_cluster(clusters, origname, cluster_directions) is not None:
            canditerm_clusters.append(check_cluster(clusters, origname, cluster_directions))
        else:
            assert origname not in alls
            canditerm_clusters.append(None)
    assert len(canditerm_clusters) == len(canditerms[0])
    canditerms = [i for i in zip(*canditerms, canditerm_clusters) if i[4] is not None]
    assert not any(i is None for i in cluster_directions.keys())
    assert not alls-set(i[3] for i in canditerms)
    assert set(i[4] for i in canditerms) == set(cluster_directions.keys())
    return canditerms, cluster_directions



if __name__ == "__main__":
    main()
