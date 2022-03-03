import pandas as pd
from matplotlib import pyplot as plt

from misc_util.logutils import setup_logging
from misc_util.pretty_print import display, pretty_print as print

from derive_conceptualspace.settings import DEFAULT_N_CPUS
from derive_conceptualspace.util.threadworker import WorkerPool
from derive_conceptualspace.pipeline import SnakeContext
from derive_conceptualspace.pipeline import load_envfiles
from derive_conceptualspace.util.result_analysis_tools import getfiles_allconfigs, make_metrics

flatten = lambda l: [item for sublist in l for item in sublist]

LAMBDA1 = 0.5

def main():
    """ In this analysis, I create a plot to underpin my claim that DESC15 likely used quadratic weighting for their kappa-score """
    #TODO: add commands/config on how to run the results fetched here!
    # Idea is: take the derrac2015.yml-configfile, add the config `kappa_weights: [quadratic, linear, None]`, check if all are run and show
    #  the command to run/schedule all (=automatically those that are not done yet). And then make a plot with all the kappas here, split by
    #  value of kappa_weights, to show that only for linear/quadratic I get sensible results
    setup_logging()
    load_envfiles("placetypes")
    configs, print_cnf = getfiles_allconfigs("featureaxes")
    configs = configs[:10]

    with WorkerPool(DEFAULT_N_CPUS, pgbar="Fetching featureaxes..") as pool:
        get_featureaxes = lambda conf: SnakeContext.loader_context(config=conf, silent=True).load("featureaxes")
        cluster_list, interrupted = pool.work(configs, get_featureaxes)

    display(f"# Going through all param-combis and checking how many κ ≥ *r*{LAMBDA1}*r* values they have:")
    detailed, displayvar = [], {}
    for conf, featureaxes in zip(configs, cluster_list):
        specials = {k: v for k, v in conf.items() if isinstance(print_cnf[k], list)}
        mets = make_metrics(featureaxes["metrics"])
        kppa = {k: len([i for i in v if i >= LAMBDA1]) for k, v in mets.items() if len([i for i in v if i >= LAMBDA1]) > 0 and "kappa" in k and k != "kappa_bin2bin"}
        displayvar[", ".join(f"{k}: {v}" for k, v in specials.items())] = kppa
        detailed.append((conf, kppa))

    display(displayvar)
    averaged = get_list_of_averageds(detailed)
    plot_perweightingalgo(averaged, detailed, LAMBDA1)


def get_list_of_averageds(detailed):
    valid_keys = set(flatten([set(i[1].keys()) for i in detailed]))
    by_kappaweights = {}
    for elem in detailed:
        by_kappaweights.setdefault(elem[0]["kappa_weights"], []).append(elem)
    list_of_dict_of_list = {}
    for key, list_of_dict in {k: [i[1] for i in v] for k, v in by_kappaweights.items()}.items():
        for di in list_of_dict:
            for k in valid_keys: #for k, v in di.items():
                list_of_dict_of_list.setdefault(key, {}).setdefault(k, []).append(di.get(k, 0))
    return {k: {k2: round(sum(v2)/len(v2)) for k2, v2 in v.items()} for k, v in list_of_dict_of_list.items()}


def plot_perweightingalgo(averaged, detailed, lambda1, do_print=True):
    # TODO what's missing:
    #  * maybe also plot standard deviation, maybe make a scatterplot for all of the param-combis instead of a simple barplot (or overlay both!)
    # TODO[i]: HAVE to add number of samples here, and SHOULD add info on what other configs (including which dataset!) are given
    # TODO[i]: Die Namen der actual kappa-funcs klingen noch ziemlich shitty
    title = f"Number of candidate-directions with κ ≥ {lambda1} per weighting-algorithm,\n averaged over {round(len(detailed)/len(averaged))} parameter-combinations each (TODOsInHere!)"
    averaged = pd.DataFrame(averaged)
    if do_print: print("\n**"+title.replace("\n", "")+":**\n\n", averaged)
    ax = averaged.plot(kind="bar", logy=True)
    ax.set_xticklabels([i._text.replace("kappa_", "").replace("_", " ") for i in ax.get_xticklabels()], ha="right", rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()

