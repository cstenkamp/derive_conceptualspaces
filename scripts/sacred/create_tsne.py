import os

from sacred import Experiment
from sacred.observers import MongoObserver

import random
import logging
from os.path import basename, join
import numpy as np
from scripts.create_siddata_dataset import load_mds, display_mds
from src.main.util.pretty_print import pretty_print as print
from src.static.settings import SID_DATA_BASE, DATA_BASE, DATA_DUMP_DIR, MONGO_URI
from src.main.load_semanticspaces import load_mds_representation, get_names
from scripts.start_autoencexplnts_pipeline import make_tsne_df
#TODO make_tsne_df shouldn't be in scripts

########################################################################################################################

ex = Experiment("Create_Siddata_tSNE")
ex.observers.append(MongoObserver(
    url=MONGO_URI,
    db_name=os.environ["MONGO_DATABASE"]))

########################################################################################################################

@ex.config
def cfg():
    mds_dimensions = 100
    data_set = "courses"
    tsne_dims = 3

@ex.automain
def my_main(mds_dimensions, data_set, tsne_dims):
    exp_inf_str = "__".join([f"{key}_{val}" for key, val in cfg().items()])
    dump_name = join(DATA_DUMP_DIR, f"tsne_{exp_inf_str}.csv")
    mds, mds_path = load_mds_representation(DATA_BASE, data_set, mds_dimensions)
    names, names_path = get_names(DATA_BASE, data_set)
    ex.add_resource(mds_path)
    ex.add_resource(names_path)
    df = make_tsne_df(mds, names, tsne_dims)
    df.to_csv(dump_name)
    ex.add_artifact(dump_name, name="tSNE")