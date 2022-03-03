import sys
from datetime import datetime
from itertools import combinations
import logging
from os.path import basename
import json

from gensim import corpora
from gensim.models import LsiModel
from tqdm import tqdm
import numpy as np
from graphviz import Digraph
import git
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine, cdist

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.misc_architecture import merge_streams
from misc_util.logutils import CustomIO
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.util.mpl_tools import actually_plot, generate_filepath

flatten = lambda l: [item for sublist in l for item in sublist]
logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################


def show_data_info(ctx):
    if ctx.get_config("DEBUG"):
        print(f"Looking at data generated in Debug-Mode for {ctx.get_config('DEBUG_N_ITEMS')} items!")

    print(f"Data lies at *b*{ctx.obj['json_persister'].in_dir}*b*")
    conf = {k: (v, [i["metadata"]["basename"] for i in ctx.p.loaded_objects.values() if k in i.get("metadata", {}).get("used_influentials", {})])
                 for k, v in ctx.p.loaded_influentials.items()} #TODO those that may differ yadda yadda
    print("Settings:\n", "\n".join(f"{k.rjust(max(len(i) for i in conf.keys()))}: *b*{str(v[0]).ljust(max(len(str(i[0])) for i in conf.values()))}*b* (used in {', '.join(v[1])})" for k, v in conf.items()), "\n")
    print("Collected Metainf:", ", ".join([f"{k}: *b*{v}*b*" for k, v in ctx.obj["json_persister"].collected_metainf().items()]))
    data_dirs = {k: v["path"].replace(ctx.obj["json_persister"].in_dir, "data_dir") for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    print("Directories:\n ", "\n  ".join(f"{k.rjust(max(len(i) for i in data_dirs))}: {v}" for k, v in data_dirs.items()))
    dependencies = {k: set([i for i in v["used_in"] if i != "this"]) for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    # figuring out when a new param was first necessary
    #TODO params that may differ in dependencies are not correctly handled yet - if DEBUG=True ONLY when showing info, it will show it already at pp_descriptions
    param_intro = {k: i.get("metadata", {}).get("used_influentials", {}) for k,i in ctx.obj["json_persister"].loaded_objects.items()}
    newparam = {}
    for key, val in {k: list(v.keys()) for k, v in param_intro.items() if v}.items():
        for elem in val:
            if elem in ctx.important_settings and elem not in flatten(newparam.values()):
                newparam.setdefault(key, []).append(elem)
    # /figuring out when a new param was first necessary
    dot = Digraph()
    for key in dependencies:
        add_txt = "\n  ".join([f"{el}: {ctx.get_config(el)}" for el in newparam.get(key, [])])
        dot.node(key, key + ("\n\n  " + add_txt if add_txt else ""))
    dot.edges([[k, e] for k, v in dependencies.items() for e in v])
    with open(generate_filepath("dependency_graph", ".dot"), "w") as wfile:
        wfile.write(dot.source) #`dot -Tpdf dependency_graph_2022-02-14_11-36-45.dot -o dependency_graph_2022-02-14_11-36-45.pdf`
    if ctx.get_config("verbose"):
        dot.render(view=True)
    commits = {k: v.get("metadata", {}).get("obj_info", {}).get("git_hash", {}).get("inner_commit") for k, v in ctx.obj["json_persister"].loaded_objects.items()}
    commit_dict = {}
    for elem in list(commits.items()):
        commit_dict.setdefault(elem[1], []).append(elem[0])
    print("Commits:\n"+("\n".join([f"{str(k)[:8].rjust(8)}: {', '.join(v)}" for k,v in commit_dict.items()])))
    # ob alle vom gleichem commit, wenn ja welcher, und die letzten 2-3 commit-messages davor
    if len([i for i in commit_dict.keys() if i is not None]) == 1:
        try:
            git_hist = list(git.Repo(".", search_parent_directories=True).iter_commits("main", max_count=20))
        except:
            pass
        else:
            commit_num = [ind for ind, i in enumerate(git_hist) if i.hexsha == list(commits.values())[0]][0]
            messages = [i.message.strip() for i in git_hist[commit_num:commit_num + 5]]
            tmp = []
            for msg in messages:
                if msg not in tmp: tmp.append(msg)
            print("Latest commit messages:\n  ", "\n   ".join(tmp))
    dates = {k: v["metadata"]["obj_info"]["date"] for k, v in ctx.obj["json_persister"].loaded_objects.items() if v.get("metadata", {}).get("obj_info", {}).get("date")}
    print("Dates:\n ", "\n  ".join(f"{k.rjust(max(len(i) for i in dates))}: {v}" for k, v in dates.items()))
    #TODO if there's PREV_RUN_INFO in the metadata (`"PREV_RUN_INFO" in ctx.obj["json_persister"].loaded_objects["featureaxes"].get("metadata", {})`) add stdout and stderr of that!
    output = {k: merge_streams(v["metadata"]["obj_info"]["stdout"], v["metadata"]["obj_info"]["stderr"], k) for k, v in ctx.obj["json_persister"].loaded_objects.items() if v.get("metadata", {}).get("obj_info", {}).get("stdout")}
    print()
    N_SPACES = 30
    while (show := input(f"Which step's output should be shown ({', '.join([k for k, v in output.items() if v])}): ").strip()) in output.keys():
        print(f"\n{'=' * N_SPACES} Showing output of **{show}** {'=' * N_SPACES}")
        print(output[show])
        print("=" * len(f"{'=' * N_SPACES} Showing output of **{show}** {'=' * N_SPACES}") + "\n")
    print()
    plots = {k: v["metadata"]["created_plots"] for k, v in ctx.obj["json_persister"].loaded_objects.items() if v.get("metadata", {}).get("created_plots")}
    while (show := input(f"Which step's plots should be shown ({', '.join([k for k, v in plots.items() if v])}): ").strip()) in plots.keys():
        print(f"Showing plots of **{show}**")
        for key, val in plots[show].items():
            print(f"  Showing plot **{key}**")
            actually_plot(json.loads(val))



########################################################################################################################
########################################################################################################################
########################################################################################################################


def rank_saldirs(pp_descriptions, embedding, featureaxes, filtered_dcm, prim_lambda, sec_lambda, metricname):
    from derive_conceptualspace.semantic_directions.create_candidate_svm import select_salient_terms
    pp_descriptions.add_embeddings(embedding.embedding_)
    decision_planes, metrics = featureaxes.values()
    existinds = {k: set(v) for k, v in filtered_dcm.term_existinds(use_index=False).items()}
    for k, v in metrics.items():
        metrics[k]["existinds"] = existinds[k]
        metrics[k]["decision_plane"] = decision_planes[k]
    n_items = len(pp_descriptions)
    clusters, directions = select_salient_terms(metrics, decision_planes, prim_lambda=prim_lambda, sec_lambda=sec_lambda, metricname=metricname)

    # TODO this is only bc in debug i set the min_existinds to 1
    metrics = {k: v for k, v in metrics.items() if len(v["existinds"]) >= 25}
    decision_planes = {k: v for k, v in decision_planes.items() if k in metrics}
    existinds = {k: v for k, v in existinds.items() if k in metrics}
    # /that

    good_candidates = dict([i for i in sorted(metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True) if
                            i[1]["accuracy"] > 0.9 and i[1]["precision"] > 0.2])
    semi_candidates = dict([i for i in sorted(metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True) if
                            i[1]["accuracy"] > 0.6 and i[1]["precision"] > 0.1 and i[1]["recall"] > 0.6 and i[
                                0] not in good_candidates])
    print()

    # jetzt will ich: Die Candidates gruppieren, die einen hohen overlap haben in welchen Texten sie vorkommen.
    # Also, wenn "a1" und "a2" in den beschreibungen von mostly den selben Kursen vorkommen, werden sie germergt.
    # TODO: was AUCH GEHT: Statt die Liste der Kurse pro Keyword anzugucken kann ich auch die Liste der Keywords pro Kurs angucken
    # und wenn da ein (dann muss es aber extremer sein) overlap ist alle keywords von kurs1 zu kurs2 hinzufügen und vice versa,
    # das wäre im Grunde die explizite version von den latent methoden LSI etc

    # combs = list(combinations(existinds.keys(), 2))
    all_overlaps = {}
    for nkey1, (key1, inds1) in enumerate(tqdm(existinds.items(), desc="Checking all overlaps")):
        n1 = len(inds1)
        for key2, inds2 in list(existinds.items())[nkey1 + 1:]:
            overlap_percentages = (n12 := len(inds1 & inds2)) / n1, n12 / len(inds2), n12 / (
                        n1 + len(inds2)), n1 / n_items, len(inds2) / n_items
            all_overlaps[(key1, key2)] = overlap_percentages
    for val in range(3):
        ar = np.array([i[val] for i in all_overlaps.values()])
        print(
            f"[{val}]: Mean Overlap of respective exist-indices: {ar[ar > 0].mean() * 100:.2f}% for those with any overlap, {ar.mean() * 100:.2f}% overall")
    merge_candidates = [[k[0], k[1]] for k, v in all_overlaps.items() if v[2] > 0.45 and max(v[3], v[4]) < 0.1]

    docfreq = {k: len(v["existinds"]) / n_items for k, v in metrics.items()}
    # TODO VISR12 have a "critique entropy", which is high for tags which seperate an entity from similar entities, isn't that useful?
    print()

########################################################################################################################
########################################################################################################################
########################################################################################################################

def run_lsi_gensim(pp_descriptions, filtered_dcm, verbose=False):
    """as in [VISR12: 4.2.1]"""
    # TODO options here:
    # * if it should filter AFTER the LSI

    if verbose:
        filtered_dcm.show_info(descriptions=pp_descriptions)
        if get_setting("DCM_QUANT_MEASURE") != "binary":
            logger.warn("VISR12 say it works best with binary!")

    filtered_dcm.add_pseudo_keyworddocs()
    dictionary = corpora.Dictionary([list(filtered_dcm.all_terms.values())])
    print("Start creating the LSA-Model with MORE topics than terms...")
    lsamodel_manytopics = LsiModel(doc_term_matrix, num_topics=len(all_terms) * 2, id2word=dictionary)
    print("Start creating the LSA-Model with FEWER topics than terms...")
    lsamodel_lesstopics = LsiModel(filtered_dcm.dtm, num_topics=len(filtered_dcm.all_terms) // 10, id2word=dictionary)
    print()
    import matplotlib.cm;
    import matplotlib.pyplot as plt
    # TODO use the mpl_tools here as well to also save plot!
    plt.imshow(lsamodel_lesstopics.get_topics()[:100, :200], vmin=lsamodel_lesstopics.get_topics().min(),
               vmax=lsamodel_lesstopics.get_topics().max(), cmap=matplotlib.cm.get_cmap("coolwarm"));
    plt.show()


def run_lsi(pp_descriptions, filtered_dcm, verbose):
    """as in [VISR12: 4.2.1]"""
    if verbose:
        filtered_dcm.show_info(descriptions=pp_descriptions)
        if get_setting("DCM_QUANT_MEASURE") != "binary":
            logger.warn("VISR12 say it works best with binary!")
    orig_len = len(filtered_dcm.dtm)
    filtered_dcm.add_pseudo_keyworddocs()
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    svd = TruncatedSVD(n_components=100, random_state=get_setting("RANDOM_SEED"))
    transformed = svd.fit_transform(filtered_dcm.as_csr().T)
    desc_psdoc_dists = cdist(transformed[:orig_len], transformed[orig_len:], "cosine")
    already_keywords = [[ind, j[0]] for ind, elem in enumerate(filtered_dcm.dtm[:orig_len]) for j in
                        elem]  # we don't gain information from those that are close but already keywords
    desc_psdoc_dists[list(zip(*already_keywords))] = np.inf
    WHICH_LOWEST = 30
    tenth_lowest = np.partition(desc_psdoc_dists.min(axis=1), WHICH_LOWEST)[
        WHICH_LOWEST]  # https://stackoverflow.com/a/43171216/5122790
    good_fits = np.where(desc_psdoc_dists.min(axis=1) < tenth_lowest)[0]
    for ndesc, keyword in zip(good_fits, np.argmin(desc_psdoc_dists[good_fits], axis=1)):
        assert not filtered_dcm.all_terms[keyword] in pp_descriptions._descriptions[ndesc]
        print(f"*b*{filtered_dcm.all_terms[keyword]}*b*", pp_descriptions._descriptions[ndesc])
    print()