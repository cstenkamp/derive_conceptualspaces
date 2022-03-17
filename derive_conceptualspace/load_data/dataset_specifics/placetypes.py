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
        candidate_min_term_count = 50, #written in DESC15
    )

    @staticmethod
    def init(ctx):
        ctx.set_config("pp_components", "none", "force[dsetclass]") #for this class, pp_components don't make any sense.
        ctx.set_config("language", "en", "force[dsetclass]") #TODO maybe automatically do this if `ctx.has_config("all_descriptions_lang") and ctx.get_config("all_descriptions_lang")`
        ctx.set_config("max_ngram", None, "force[dsetclass]") #the data we have is a preprocessed BoW we cannot get the ngrams from

    @staticmethod
    def preprocess_raw_file(jsn, pp_components):
        print("Ignoring all PP-Components for this dataset!")
        return jsn

    CATNAMES = {
        "Geonames": {1: "stream,lake", 2: "parks,area", 3: "road,railroad", 4: "spot,building,farm", 5: "mountain,hill,rock", 6: "undersea", 7: "forest,heath"}, #can recover from http://www.geonames.org/export/codes.html
        "Foursquare": {1: "Arts&Entertainment", 2: "College&University", 3: "Food", 4: "Professional&Other", 5: "NightlifeSpots", 6: "GreatOutdoors", 7: "Shops&Services", 8:"Travel&Transport", 9:"Residences"}, #https://web.archive.org/web/20140625051659/http://aboutfoursquare.com/foursquare-categories/
    }


def get_classes(data_base, what):
    assert what in ["Foursquare", "Geonames", "CYC"] or all(i in ["Foursquare", "Geonames", "CYC"] for i in what)
    names = get_names(data_base, "places")
    alls = {}
    for wha in what if isinstance(what, list) else [what]:
        if wha in ["Foursquare", "Geonames"]:
            classes = {k: int(v) for k, v in dict(np.loadtxt(join(data_base, "places", wha+"Classes.txt"), dtype=str, delimiter="\t")).items()}
        else:
            raise NotImplementedError("TODO: CYC")
        alls[wha] = {name: classes.get(name) for name in names if name in classes}
    return alls if isinstance(what, list) else alls[what]


def get_candidateterms(data_base, data_set, n_dims, **kwargs):
    dir = join(data_base, data_set, f"d{n_dims}", "DirectionsHeal")
    vecnames = [i for i in os.listdir(dir) if i.endswith(".vector")]
    vectors = [np.loadtxt(join(dir, i)) for i in vecnames]
    vecnames = [i[:-len(".vector")] for i in vecnames]
    return vecnames, vectors


# if __name__ == "__main__":
#     print(get_classes(SPACES_DATA_BASE, "Genres"))