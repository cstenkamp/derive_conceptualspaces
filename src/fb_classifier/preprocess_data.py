from os.path import join, isfile, dirname, splitext
import os
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.fb_classifier.util.load_data import load_data
import src.static.classifier_config
from src.static.classifier_config import PP_TRAIN_PERCENTAGE
from src.fb_classifier.util.misc import read_config, write_config


def get_pp_config():
    '''gets all configs from main.config that start with PP_ as dict.'''
    pp_config = {i: eval('src.static.classifier_config.' + i) for i in dir(src.static.classifier_config) if i.startswith('PP_')}
    pp_config_str = '\n'.join(f'{key}: {val}' for key, val in pp_config.items())
    return pp_config_str


def create_traintest(sourcefile):
    all_sources = dict(zip(["train", "test"], train_test_split(pd.read_csv(sourcefile), train_size=PP_TRAIN_PERCENTAGE)))
    sources = {}
    for source, cont in all_sources.items():
        cont.to_csv(join(dirname(sourcefile), splitext(sourcefile)[0]+"_"+source + ".csv"))
        sources[source] = join(dirname(sourcefile), splitext(sourcefile)[0]+"_"+source + ".csv")
    return sources


def preprocess_data(sources, dest_path: str, force_overwrite: bool = False): #sources is dict or str
    '''Preprocesses the data. Only runs if it's not there yet, force_overwrite==True, or preprocessing-configs changed (see get_config())'''
    os.makedirs(dest_path, exist_ok=True)
    if isinstance(sources, str):
        all_sources = dict(zip(["train", "test"], train_test_split(pd.read_csv(sources), train_size=PP_TRAIN_PERCENTAGE)))
        sources = {}
        for source, cont in all_sources.items():
            cont.to_csv(join(dest_path, source+".csv"))
            sources[source] = join(dest_path, source+".csv")
    if force_overwrite or get_pp_config() != read_config(dest_path):
        for file in os.listdir(dest_path):
            os.remove(join(dest_path, file))
    if not all(isfile(join(dest_path, fname+'.csv')) for fname in sources.keys()):
        print("Preprocessing the raw data...")
        data = load_data(sources)
        all_vals = set()
        for key, val in data.items():
            val = make_classifier_class(key, val)
            val.to_csv(join(dest_path, key+'.csv'))
            all_vals.update(set(val.values))
        write_config(dest_path, get_pp_config())
        with open(join(dest_path, "meta"), "w") as wfile:
            wfile.write(f"classes: {len(all_vals)}")
    return {key: join(dest_path, key+'.csv') for key in sources.keys()}


def make_classifier_class(dsetname, dset):
    new_dset = {}
    for key, val in dset.items():
        if len(splits := val.split(".")) == 2 and splits[0].isnumeric() and splits[1].replace("_", "").isnumeric():
            while len(splits[0]) > 0 and splits[0].startswith("0"):
                splits[0] = splits[0][1:]
            if len(splits[0]) > 0:
                new_dset[key] = splits[0]
        elif len(splits := val.split(".")) == 2 and splits[0].isnumeric() and splits[1][:-1].isnumeric() and splits[1][-1] in "abcdefg":
            new_dset[key] = splits[0]
        elif val.startswith("FB") and val[2:val.find(" ")].isnumeric():
            res = val[2:val.find(" ") if val.find(" ") > 0 else None]
            if res.isnumeric():
                while len(res) > 0 and res.startswith("0"):
                    res = res[1:]
                if len(res) > 0:
                    new_dset[key] = res
        else:
            new_dset[key] = "other"
    usables = {k:int(v)-1 for k,v in new_dset.items() if v != "other" and int(v) <=10}
    print(f"{dsetname}: dropped {len(new_dset)-len(usables)} courses as they had no unamiguous Fachbereich")
    counter = dict(sorted(Counter([i+1 for i in usables.values()]).items(), key=lambda x:int(x[0])))
    plt.bar(*list(zip(*counter.items())))
    plt.title(f"Number of Courses per Fachbereich ({dsetname}-dataset)")
    plt.show() #TODO show in a notebook
    return pd.Series(usables)