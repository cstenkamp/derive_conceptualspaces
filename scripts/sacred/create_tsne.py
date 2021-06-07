import os

from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd
from sklearn.manifold import TSNE

from os.path import join
from src.static.settings import DATA_BASE, DATA_DUMP_DIR, MONGO_URI
from src.main.load_data.load_semanticspaces import load_mds_representation, get_names

########################################################################################################################

ex = Experiment("Create_tSNE")
ex.observers.append(MongoObserver(url=MONGO_URI, db_name=os.environ["MONGO_DATABASE"]))

########################################################################################################################

from src.static.settings import DATA_SET

@ex.config
def cfg():
    mds_dimensions = 100
    data_set = "movies"
    tsne_dims = 2


def make_tsne_df(mds, names, n_dims=3):
    if hasattr(mds, "embedding_"):
        mds = mds.embedding_
    df = pd.DataFrame(mds, columns=[f"dim{i}" for i in range(mds.shape[1])])
    df["Name"] = names
    tsne_res = TSNE(n_components=n_dims).fit_transform(mds)
    # TODO PCA geht genauso gut, dann gibt's auch explained variance, see https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    for i in range(n_dims):
        df[f"tsne_{i + 1}"] = tsne_res[:, i]
    return df

@ex.automain
def main(mds_dimensions, data_set, tsne_dims):
    exp_inf_str = "__".join([f"{key}_{val}" for key, val in cfg().items()])
    dump_name = join(DATA_DUMP_DIR, f"tsne_{exp_inf_str}.csv")
    mds, mds_path = load_mds_representation(DATA_BASE, data_set, mds_dimensions)
    names, names_path = get_names(DATA_BASE, data_set)
    ex.add_resource(mds_path)
    ex.add_resource(names_path)
    df = make_tsne_df(mds, names, tsne_dims)
    df.to_csv(dump_name)
    ex.add_artifact(dump_name, name="tSNE")