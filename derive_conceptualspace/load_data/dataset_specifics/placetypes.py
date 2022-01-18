from os.path import join, isdir
import os

import numpy as np

from derive_conceptualspace.load_data.load_semanticspaces import get_names


def get_classes(data_base, what):
    #broken for ratings (files are all ~half the number of movies), any reason why?
    assert what in ["Foursquare", "Geonames", "CYC"]
    names = get_names(data_base, "places")
    if what in ["Foursquare", "Geonames"]:
        classes = {k: int(v) for k, v in dict(np.loadtxt(join(data_base, "places", what+"Classes.txt"), dtype=str, delimiter="\t")).items()}
    else:
        raise NotImplementedError("TODO: CYC")
    classified = {name: classes.get(name) for name in names if name in classes}
    return classified


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    dir = join(data_base, data_set, f"d{n_dims}", "DirectionsHeal")
    vecnames = [i for i in os.listdir(dir) if i.endswith(".vector")]
    vectors = [np.loadtxt(join(dir, i)) for i in vecnames]
    vecnames = [i[:-len(".vector")] for i in vecnames]
    return vecnames, vectors


# if __name__ == "__main__":
#     print(get_classes(SPACES_DATA_BASE, "Genres"))