from os.path import join, isdir
import os

import numpy as np

from derive_conceptualspace.load_data.load_semanticspaces import get_names

ORDER = ['Musical', 'Music', 'Documentary', 'Western', 'Animation', 'War', 'History', 'Sci-Fi', 'Horror', 'Sport', 'Biography', 'Film-Noir', 'News', 'Fantasy', 'Adult', 'Crime', 'Thriller', 'Comedy', 'Romance', 'Action', 'Mystery', 'Adventure', 'Drama', 'Family', 'Short']
#this order is supposed to be roughly sorted by informativeness, such that a movie that is both "Musical" and "Family", if only one label is supposed to be picked, will rather be the more informative "Musical"

def get_classes(data_base, what):
    #broken for ratings (files are all ~half the number of movies), any reason why?
    assert what in ["Genres", "Keywords", "Ratings"]
    names = get_names(data_base, "movies")
    class_dirs = {i[len("classes"):]: join(data_base, "movies", i) for i in os.listdir(join(data_base, "movies")) if isdir(join(data_base, "movies", i)) and i.startswith("classes")}
    cls_dir = class_dirs[what]
    classes = {i[len("class-"):]: list(np.loadtxt(join(cls_dir,i), delimiter="\n", dtype=bool)) for i in os.listdir(cls_dir)}
    classes = {n: [key for key, val in classes.items() if val[i]] for i, n in enumerate(names)}
    if what == "Genres":
        classes = {key: sorted(val, key=lambda x:ORDER.index(x)) for key, val in classes.items()}
    return classes


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    dir = join(data_base, data_set, f"d{n_dims}", "DirectionsHeal")
    vecnames = [i for i in os.listdir(dir) if i.endswith(".vector")]
    vectors = [np.loadtxt(join(dir, i)) for i in vecnames]
    vecnames = [i[:-len(".vector")] for i in vecnames]
    vecnames2 = [list(zip(*[j.split("_") for j in i.split(" ")])) for i in vecnames]
    words, poss = [" ".join(i[0]) for i in vecnames2], [" ".join(i[1]) if len(i) > 1 else None for i in vecnames2]
    return words, poss, vectors, vecnames


# if __name__ == "__main__":
#     print(get_classes(SPACES_DATA_BASE, "Genres"))