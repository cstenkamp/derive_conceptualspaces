from os.path import join, isdir
import os

import numpy as np

from derive_conceptualspace.load_data.load_semanticspaces import get_names
from derive_conceptualspace.load_data.dataset_specifics import BaseDataset


class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.json",
        all_descriptions_lang = "en",
        preprocessed_bow = True,
    )

    @staticmethod
    def init(ctx):
        ctx.set_config("pp_components", "none", "force") #for this class, pp_components don't make any sense.
        ctx.set_config("language", "en", "force")

    @staticmethod
    def preprocess_raw_file(jsn, pp_components):
        print("Ignoring all PP-Components for this dataset!")
        return jsn


def get_classes(data_base, what):
    assert what in ["Foursquare", "Geonames", "CYC"] or all(i in ["Foursquare", "Geonames", "CYC"] for i in what)
    names = get_names(data_base, "places")
    if not isinstance(what, list):
        what = [what]
    alls = {}
    for wha in what:
        if wha in ["Foursquare", "Geonames"]:
            classes = {k: int(v) for k, v in dict(np.loadtxt(join(data_base, "places", wha+"Classes.txt"), dtype=str, delimiter="\t")).items()}
        else:
            raise NotImplementedError("TODO: CYC")
        alls[wha] = {name: classes.get(name) for name in names if name in classes}
    return alls


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    dir = join(data_base, data_set, f"d{n_dims}", "DirectionsHeal")
    vecnames = [i for i in os.listdir(dir) if i.endswith(".vector")]
    vectors = [np.loadtxt(join(dir, i)) for i in vecnames]
    vecnames = [i[:-len(".vector")] for i in vecnames]
    return vecnames, vectors


# if __name__ == "__main__":
#     print(get_classes(SPACES_DATA_BASE, "Genres"))