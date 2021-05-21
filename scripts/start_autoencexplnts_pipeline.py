import numpy as np
import logging
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from src.main.load_semanticspaces import load_ppmi_weighted_feature_vectors, load_mds_representation, get_names
from src.static.settings import DATA_BASE, DATA_SET, MDS_DIMENSIONS
from src.main.measures import simple_similiarity, between_a, between_b_inv
from src.main.util.pretty_print import pretty_print as print, fmt
from src.main.util.logging import setup_logging
from src.test.test_semanticspaces_measures import find_betweenness_position
from scripts.create_siddata_dataset import get_data

SOME_IDS = {"Computer Vision": 4155, "Computergrafik": 547, "Computergrafikpraktikum": 453, "Machine Learning": 1685, "Rechnernetzepraktikum": 1921}
#[(ni, i) for ni, i in enumerate(tmp2) if "Machine Learning" in i]

def get_descriptions():
    from src.static.settings import SID_DATA_BASE
    from scripts.create_siddata_dataset import load_mds
    names, descriptions, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{MDS_DIMENSIONS}.json")) #TODO #PRECOMMIT comment out other line
    return dict(zip(names, descriptions))

def make_tsne_df(n_dims=3):
    # TODO argument um gleich viele Kurse pro FB zu plotten
    tmp = load_mds_representation(DATA_BASE, DATA_SET, MDS_DIMENSIONS)
    tmp2 = get_names(DATA_BASE, DATA_SET)
    # name_mds = dict(zip(tmp2, tmp))
    name_number = get_data().set_index("Name")["VeranstaltungsNummer"].to_dict()
    fachbereich_per_course = {k: int(v.split(".", 1)[0]) for k, v in name_number.items() if
                              v.split(".", 1)[0].isdigit() and int(v.split(".", 1)[0]) <= 10}  # There are 10 FBs
    fb_mapper = {1: "Sozial,Kultur,Kunst", 3: "Theologie,Lehramt,Musik", 4: "Physik", 5: "Bio,Chemie", 6: "Mathe,Info",
                 7: "Sprache,Literatur", 8: "Humanwiss", 9: "Wiwi", 10: "Rechtswiss"}
    df = pd.DataFrame(tmp, columns=[f"dim{i}" for i in range(tmp.shape[1])])
    df["Name"] = tmp2
    df["FB"] = [fachbereich_per_course.get(name, 0) for name in df["Name"]]
    df["FB_long"] = [fb_mapper.get(i, "Unknown") for i in df["FB"]]
    tsne_res = TSNE(n_components=n_dims).fit_transform(tmp)
    # TODO PCA geht genauso gut, dann gibt's auch explained variance, see https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    for i in range(n_dims):
        df[f"tsne_{i + 1}"] = tsne_res[:, i]
    return df

def main():
    setup_logging("DEBUG")
    # show_betwennesses()


def show_betwennesses():
    tmp = load_mds_representation(DATA_BASE, DATA_SET, MDS_DIMENSIONS)
    tmp2 = get_names(DATA_BASE, DATA_SET)
    name_mds = dict(zip(tmp2, tmp))
    candidates = [("Computergrafik", "Computer Vision", "Machine Learning"), ("Rechnernetzepraktikum", "Computergrafik", "Computergrafikpraktikum")]
    descriptions = get_descriptions()
    candidates = [tuple(tmp2[SOME_IDS[j]] for j in i) for i in candidates]
    for first, second, third in candidates:
        find_betweenness_position(name_mds, first, second, third, between_a, descriptions=descriptions)



if __name__ == '__main__':
    main()
