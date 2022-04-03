import os
from functools import partial
from os.path import join, dirname, splitext
from parse import parse
import math
import warnings
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from derive_conceptualspace.evaluate.shallow_trees import classify_shallowtree
from derive_conceptualspace.pipeline import SnakeContext
from derive_conceptualspace.util.threadworker import WorkerPool
from misc_util.pretty_print import display

from derive_conceptualspace import settings
from derive_conceptualspace.settings import standardize_config_name, DEFAULT_N_CPUS
from derive_conceptualspace.util.jsonloadstore import get_file_config


def getfiles_allconfigs(basename, dataset=None, base_dir=None, ext=".json", only_nondebug=True, parse_all=False, verbose=True):
    dataset = dataset or os.environ["MA_DATASET"]
    base_dir = base_dir or os.environ["MA_BASE_DIR"]
    candidates = [join(path, name)[len(base_dir)+1:] for path, subdirs, files in os.walk(base_dir) for
                  name in files if splitext(name)[0].startswith(basename) and splitext(name)[1]==ext]
    candidates = [i for i in candidates if i.startswith(dataset) and not "backup" in i.lower() and not "interrupted" in i.lower()]
    if (leftovers := [cand for cand in candidates if not parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]+[basename+ext]), cand) and
            (not (prs := parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]), dirname(cand))) or
            (not only_nondebug or not prs.named.get("debug", False) in ["False", False]))]):
        if verbose:
            warnings.warn("There are files that won't be considered here: \n    "+"\n    ".join(leftovers))
        candidates = [i for i in candidates if i not in leftovers]
    if parse_all:
        configs = [get_file_config(os.environ["MA_BASE_DIR"], cand, re.findall(r'{(.*?)}', "".join(settings.DIR_STRUCT))) for cand in candidates]
    else:
        filename_confs = [i for i in re.findall(r'{(.*?)}', os.sep.join(settings.DIR_STRUCT))]
        getconf = lambda cand: get_file_config(os.environ["MA_BASE_DIR"], cand, re.findall(r'{(.*?)}', "".join(settings.DIR_STRUCT)))
        trnsl = {standardize_config_name(i): i for i in filename_confs}
        order = {{v2: k2 for k2,v2 in trnsl.items()}[v]: k for k, v in enumerate(filename_confs)} #ensure later settings come up later
        configs = [{trnsl[k]: v for k, v in sorted([(k2,v2) for k2,v2 in getconf(cand).items() if k2 in trnsl], key=lambda x:order[x[0]])} for cand in candidates]
        # configs = [parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]+[basename+ext]), cand).named for cand in candidates if parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]+[basename+ext]), cand)] \
        #          + [prs.named for cand in candidates if (prs := parse(os.sep.join(settings.DIR_STRUCT[:cand.count(os.sep)]+[basename+"_"+("_".join(re.findall(r'({.*?})', settings.DIR_STRUCT[cand.count(os.sep)+1])))+ext]), cand))]
        #          #the second summand is for the cases where where not all configs to justify a new dir are there so the rest of the configs overflow to the filename
    if only_nondebug:
        configs = [i for i in configs if (i.get("debug") or i.get("DEBUG")) not in ["True", True]]
    if not configs:
        raise FileNotFoundError("There are no usable configs!")
    print_cnf = {k: list(set(dic[k] for dic in configs)) for k in configs[0]}
    print_cnf = {k: [str(i) for i in sorted([int(j) for j in v])] if all(str(k2).isnumeric() for k2 in v) else sorted([str(j) for j in v]) for k, v in print_cnf.items()}
    configs = sorted(configs, key=lambda elem: sum([print_cnf[k].index(str(v))*(10**(len(elem)-n)) for n, (k, v) in enumerate(elem.items())]))
    print_cnf = {k: v[0] if len(v) == 1 else v for k, v in print_cnf.items()}
    if verbose:
        display(f"There are {len(configs)} different parameter-combis for dataset *b*{os.environ['MA_DATASET']}*b*:")
        display(print_cnf)
    return configs, print_cnf



def make_metrics(metlist):
    metrics =  {k: None for k in list(metlist.values())[0].keys()}
    for met in metrics.keys():
        vals = [i[met] for i in metlist.values()]
        vals = [abs(i) for i in vals]
        #vals = (np.digitize(vals, [i/10 for i in arange(11)])-1)/10
        metrics[met] = vals
    return metrics


def display_metrics(metlist):
    metrics = make_metrics(metlist)
    w, h = math.ceil(math.sqrt(len(metrics))), math.ceil(math.sqrt(len(metrics)))
    for n, (met, vals) in enumerate(metrics.items()):
        plt.subplot(w, h, n+1)
        hist = np.histogram(vals, bins=[i/10 for i in range(11)]+[max(vals+[1.1])])
        make_logscale = max(hist[0])/len(vals) > 0.6
        plt.title(met +(" (log)" if make_logscale else ""))
        plt.hist(vals, bins=[i/10 for i in range(11)]+[max(vals+[1.1])], log=make_logscale, color="red" if make_logscale else "blue")
    plt.tight_layout()
    plt.show()


def show_lambda_elements(metlist, lambda1=0.5, lambda2=0.1):
    for met in list(list(metlist.values())[0].keys()):
        if "kappa" in met and not "bin2bin" in met:
            vals = [i[met] for i in metlist.values()]
            t1 = len([i for i in vals if i >= lambda1])
            t2 = len([i for i in vals if i >= lambda2]) - t1
            if t1:
                print(f" {met}: T^{lambda1}: {t1}, T^{lambda2}: {t2}")
                print(f" In T^{lambda1}: {', '.join([k for k, v in metlist.items() if v[met] > lambda1])}")


def highlight_nonzero_max(data):
    #df.style.apply(highlight_nonzero_max, axis=0), https://stackoverflow.com/a/62639983/5122790
    #df.style.highlight_max(color='lightgreen', axis=0)
    # return [f'font-weight: bold' if v == data.max() and v > 0 else '' for v in data]
    return [f'background-color: lightgreen' if v == data.max() and v > 0 else '' for v in data]


def df_to_latex(df, styler, resizebox=True, bold_keys=True, rotate="45", rotate_index=False, caption=None):
    rotate = str(rotate) if rotate is not False else False
    df = df.copy()
    if bold_keys:
        if df.index.names != [None]:
            indexnames = ["\\textbf{"+i+"}" for i in df.index.names]
            if rotate and rotate_index: indexnames = ["\\rotatebox{"+rotate+"}{"+i+"}" for i in indexnames]
            df.index = pd.MultiIndex.from_tuples([["\\textbf{"+str(j)+"}" for j in i] for i in df.index], names=indexnames)
        df.columns = ["\\textbf{"+(str(i[1]) if isinstance(i, (list, tuple)) else i)+"}" for i in df.columns] #i[1] if it's a named index
    res = styler(df)
    if rotate: res.applymap_index(lambda v: "rotatebox:{"+rotate+"}--rwrap--latex;", axis=1)
    txt = res.to_latex(convert_css=True, clines="skip-last;index", multirow_align="t", hrules=True, siunitx=False, caption=caption)
    txt = [i for i in txt.split("\n") if i != "\\thtop"]
    if resizebox: txt = [txt[0]]+["\\resizebox{\\textwidth}{!}{%"]+txt[1:-2]+["}"]+txt[-2:]
    for nrow in range(len(txt)-1):
        if txt[nrow].startswith("\\cline") and txt[nrow+1] == "\\bottomrule":
            txt[nrow] = ""
    txt = "\n".join([i for i in txt if i]).replace("nan", "-")
    txt = txt.replace("≥", "$\\geq$").replace("%", "\%")
    return txt



SHORTEN_DICT = {
    "kappa": "k",
    "dense": "d",
    "rank2rank": "r2r",
    "count2rank": "c2r",
    "bin2bin": "b2b",
    "f_one": "f1",
    "digitized": "dig",
    "_onlypos": "+"
}
def shorten_met(met, reverse=False):
    for k, v in ({v2: k2 for k2, v2 in SHORTEN_DICT.items()} if reverse else SHORTEN_DICT).items():
        met = met.replace(k, v)
    return met


def get_best_conf(classes, nprocs=DEFAULT_N_CPUS-1, verbose=True, return_all=False, **kwargs): #kwargs can be: balance_classes, test_percentage_crossval, dt_depth, one_vs_rest, metric
    configs, print_cnf = getfiles_allconfigs("clusters", verbose=True)
    def get_tree_perf(conf, print_cnf):
        ctx = SnakeContext.loader_context(config=conf, silent=True, warn_filters=["DifferentFileWarning"])
        clusters, embedding, descriptions = ctx.load("clusters", "embedding", "pp_descriptions")
        res = classify_shallowtree(clusters, embedding, descriptions, ctx.obj["dataset_class"], classes=classes,
                                   return_features=False, shutup=True, clus_rep_algo="top_1", **kwargs)
        cnf = tuple(v for k, v in conf.items() if isinstance(print_cnf[k], list))
        return cnf, res
    with WorkerPool(nprocs, pgbar="Getting Best-Performing Config") as pool:
        perconf_list, interrupted = pool.work(configs, partial(get_tree_perf, print_cnf=print_cnf))
    best = max(perconf_list, key=lambda x:x[1])
    if verbose:
        if len(set(dict(perconf_list))-{tuple()}) > 0:
            table = pd.DataFrame(dict(perconf_list).values(), index=dict(perconf_list).keys(), columns=["Accuracy"]).unstack(level=[0,1,2])
            display(table)
        print(f"Best Accuracy: {best[1]:.2%}")
    if return_all:
        return pd.DataFrame(dict(perconf_list), columns=pd.MultiIndex.from_arrays(list(zip(*dict(perconf_list).keys())), names=[k for k,v in print_cnf.items() if isinstance(v, list)]), index=["accuracy"])
    res = ({**{k: v for k, v in print_cnf.items() if not isinstance(v, list)}, **dict(zip([k for k, v in print_cnf.items() if isinstance(v, list)],best[0]))}, best[1])
    return res