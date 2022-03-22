from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd
from sklearn.manifold import TSNE

from os.path import join

import os
from dotenv.main import load_dotenv
load_dotenv("../../../../docker/.env")
assert os.getenv("MONGO_INITDB_ROOT_USERNAME") and os.getenv("MONGO_INITDB_ROOT_PASSWORD")


from fb_classifier.settings import MONGO_URI
from derive_conceptualspace.load_data.load_semanticspaces import load_mds_representation, get_names
from derive_conceptualspace.settings import DEFAULT_BASE_DIR
os.makedirs(DEFAULT_BASE_DIR, exist_ok=True)

DATA_BASE = "/home/chris/Documents/UNI_neu/Masterarbeit/data_new/semanticspaces/"

########################################################################################################################

ex = Experiment("Create_tSNE")
ex.observers.append(MongoObserver(url=MONGO_URI, db_name=os.environ["MONGO_DATABASE"]))

########################################################################################################################


@ex.config
def cfg():
    mds_dimensions = 100
    data_set = "places"
    tsne_dims = 3


def make_tsne_df(mds, names, n_dims=3):
    if hasattr(mds, "embedding_"):
        mds = mds.embedding_
    df = pd.DataFrame(mds, columns=[f"dim{i}" for i in range(mds.shape[1])])
    df["Name"] = names
    tsne_res = TSNE(n_components=n_dims, metric="cosine").fit_transform(mds)
    # TODO PCA geht genauso gut, dann gibt's auch explained variance, see https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    for i in range(n_dims):
        df[f"tsne_{i + 1}"] = tsne_res[:, i]
    return df

@ex.automain
def main(mds_dimensions, data_set, tsne_dims):
    exp_inf_str = "__".join([f"{key}_{val}" for key, val in cfg().items()])
    dump_name = join(DEFAULT_BASE_DIR, f"tsne_{exp_inf_str}.csv")
    path_obj = []
    mds = load_mds_representation(DATA_BASE, data_set, mds_dimensions, fname_out=path_obj)
    names = get_names(DATA_BASE, data_set, fname_out=path_obj)
    ex.add_resource(path_obj[0])
    ex.add_resource(path_obj[1])
    df = make_tsne_df(mds, names, tsne_dims)
    df.to_csv(dump_name)
    ex.add_artifact(dump_name, name="tSNE")