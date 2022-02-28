import os
from os.path import join, dirname, splitext
from parse import parse
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt

from derive_conceptualspace import settings
from misc_util.pretty_print import display


def getfiles_allconfigs(basename, dataset=None, base_dir=None, ext=".json", only_nondebug=True, verbose=True):
    dataset = dataset or os.environ["MA_DATASET"]
    base_dir = base_dir or os.environ["MA_BASE_DIR"]
    candidates = [join(path, name)[len(base_dir)+1:] for path, subdirs, files in os.walk(base_dir) for
                  name in files if splitext(name)[0].startswith(basename) and splitext(name)[1]==ext]
    candidates = [i for i in candidates if i.startswith(dataset) and not "backup" in i.lower()]
    if (leftovers := [cand for cand in candidates if not parse(os.sep.join(settings.DIR_STRUCT+[basename+ext]), cand) and
                                   (not parse(os.sep.join(settings.DIR_STRUCT), dirname(cand)) or
                                     (not only_nondebug or parse(os.sep.join(settings.DIR_STRUCT), dirname(cand)).named.get("debug", False) in ["False", False]))]):
        if verbose:
            warnings.warn("There are files that won't be considered here: \n    "+"\n    ".join(leftovers))
        candidates = [i for i in candidates if i not in leftovers]
    configs = [parse(os.sep.join(settings.DIR_STRUCT+[basename+ext]), cand).named for cand in candidates if parse(os.sep.join(settings.DIR_STRUCT+[basename+ext]), cand)]
    if only_nondebug:
        configs = [i for i in configs if i["debug"] not in ["True", True]]
    print_cnf = {k: list(set(dic[k] for dic in configs)) for k in configs[0]}
    print_cnf = {k: [str(i) for i in sorted([int(j) for j in v])] if all(k.isnumeric() for k in v) else sorted(v) for k, v in print_cnf.items()}
    configs = sorted(configs, key=lambda elem: sum([print_cnf[k].index(v)*(10**(len(elem)-n)) for n, (k, v) in enumerate(elem.items())]))
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