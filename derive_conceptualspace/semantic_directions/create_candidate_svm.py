from itertools import repeat
from multiprocessing.pool import ThreadPool
import warnings
from textwrap import shorten
import random

from tqdm import tqdm
import sklearn.svm
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import rankdata
import pandas as pd

from derive_conceptualspace.settings import get_setting, IS_INTERACTIVE, get_ncpu
from derive_conceptualspace.util.base_changer import NDPlane, ThreeDPlane
from derive_conceptualspace.util.threadworker import WorkerPool
from derive_conceptualspace.util.threedfigure import ThreeDFigure
from misc_util.pretty_print import pretty_print as print

norm = lambda vec: vec/np.linalg.norm(vec)
vec_cos = lambda v1, v2: np.arccos(np.clip(np.dot(norm(v1), norm(v2)), -1.0, 1.0))  #https://stackoverflow.com/a/13849249/5122790


def create_candidate_svms(dcm, embedding, descriptions, verbose):
    decision_planes = {}
    metrics = {}
    compareto_ranking = get_setting("classifier_compareto_ranking")
    if dcm.quant_name != compareto_ranking and dcm.quant_name == "count":
        dcm = dcm.apply_quant(compareto_ranking, descriptions=descriptions, verbose=verbose)
    elif dcm.quant_name != compareto_ranking:
        # Ensure that regardless of quant_measure `np.array(quants, dtype=bool)` are correct binary classification labels
        raise NotImplementedError()
    terms = list(dcm.all_terms.values())
    if get_setting("DEBUG"):
        # terms = terms[:get_setting("DEBUG_N_ITEMS")]
        assert all(i in terms for i in ['nature', 'ceiling', 'engine', 'athlete', 'seafood', 'shadows', 'skyscrapers', 'b737', 'monument', 'baby', 'sign', 'marine', 'iowa', 'field', 'buy', 'military', 'lounge', 'factory', 'road', 'education', '13thcentury', 'people', 'wait', 'travel', 'tunnel', 'treno', 'wings', 'hot', 'background', 'vintage', 'farmhouse', 'technology', 'building', 'horror', 'realestate', 'crane', 'slipway', 'ruin', 'national', 'morze'])
        terms = ['nature', 'ceiling', 'engine', 'athlete', 'seafood', 'shadows', 'skyscrapers', 'b737', 'monument', 'baby', 'sign', 'marine', 'iowa', 'field', 'buy', 'military', 'lounge', 'factory', 'road', 'education', '13thcentury', 'people', 'wait', 'travel', 'tunnel', 'treno', 'wings', 'hot', 'background', 'vintage', 'farmhouse', 'technology', 'building', 'horror', 'realestate', 'crane', 'slipway', 'ruin', 'national', 'morze']
        assert len([i for i in descriptions._descriptions if 'nature' in i]) == len([i for i in dcm.term_quants('nature') if i > 0])
        #TODO #PRECOMMIT #FIXPRECOMMIT remove me!
    assert all(len([i for i in descriptions._descriptions if term in i]) == len([i for i in dcm.term_quants(term) if i > 0]) for term in random.sample(terms, 5))
    if get_ncpu() == 1:
        quants_s = [dcm.term_quants(term) for term in tqdm(terms, desc="Counting Terms")]
        for term, quants in tqdm(zip(terms, quants_s), desc="Creating Candidate SVMs", total=len(terms)):
            cand_mets, decision_plane, term = create_candidate_svm(embedding.embedding_, term, quants, quant_name=dcm.quant_name)
            metrics[term] = cand_mets
            decision_planes[term] = decision_plane
    else:
        print(f"Starting Multiprocessed with {get_ncpu()} CPUs")
        with WorkerPool(get_ncpu(), dcm, pgbar="Counting Terms") as pool:
            quants_s = pool.work(list(terms), lambda dcm, term: dcm.term_quants(term))
        with tqdm(total=len(quants_s), desc="Creating Candidate SVMs") as pgbar, ThreadPool(get_ncpu()) as p:
            res = p.starmap(create_candidate_svm, zip(repeat(embedding.embedding_), terms, quants_s, repeat(False), repeat(None), repeat(dcm.quant_name), repeat(pgbar)))
        for cand_mets, decision_plane, term in res:
            metrics[term] = cand_mets
            decision_planes[term] = decision_plane
    if (didnt_converge := len([1 for i in metrics.values() if not i["did_converge"]])):
        warnings.warn(f"{didnt_converge} of the {len(metrics)} SVMs did not converge!", sklearn.exceptions.ConvergenceWarning)
    if verbose:
        df = pd.DataFrame(metrics).T
        df.columns = df.columns.str.replace("kappa", "k").str.replace("rank2rank", "r2r").str.replace("bin2bin", "b2b").str.replace("f_one", "f1").str.replace("digitized", "dig")
        for metricname in df.columns:
            print(f"\nAverage *r*{metricname}*r*: {df[metricname].mean():.5f}")
            with pd.option_context('display.max_rows', 11, 'display.max_columns', 20, 'display.expand_frame_repr', False, 'display.max_colwidth', 20, 'display.float_format', '{:.4f}'.format):
                print(str(df.sort_values(by=metricname, ascending=False)[:10]).replace(metricname, f"*r*{metricname}*r*"))
        if embedding.embedding_.shape[1] == 3 and IS_INTERACTIVE:
            best_elem = max(metrics.items(), key=lambda x:x[1]["f_one"])
            create_candidate_svm(embedding.embedding_, best_elem[0], dcm.term_quants(best_elem[0]), quant_name=dcm.quant_name, plot_svm=True, descriptions=descriptions)
            while (another := input("Another one to display: ").strip()) != "":
                if "," in another:
                    highlight = [i.strip() for i in another.split(",")[1:]]
                    another = another.split(",")[0].strip()
                else:
                    highlight = []
                create_candidate_svm(embedding.embedding_, another, dcm.term_quants(another), quant_name=dcm.quant_name, plot_svm=True, descriptions=descriptions, highlight=highlight)
    # clusters, cluster_directions = select_salient_terms(sorted_kappa, decision_planes, prim_lambda, sec_lambda)
    return decision_planes, metrics

class Comparer():
    def __init__(self, decision_planes, compare_fn):
        self.decision_planes = decision_planes
        self.already_compared = {}
        self.compare_fn = compare_fn
    def __call__(self, v1, v2):
        if v1+","+v2 not in self.already_compared:
            if v2+","+v1 in self.already_compared:
                return self.already_compared[v2+","+v1]
            self.already_compared[v1+","+v2] = self.compare_fn(self.decision_planes[v1].normal, self.decision_planes[v2].normal)
        return self.already_compared[v1+","+v2]


def select_salient_terms(metrics, decision_planes, prim_lambda, sec_lambda, metricname):
    #TODO waitwaitwait. Am I 100% sure that the intercepts of the decision_planes are irrelevant?!
    print(f"Calculated Metrics: {list(list(metrics.values())[0].keys())}")
    print(f"Lambda1: {prim_lambda}, Lambda2: {sec_lambda}, compareto-metric: {metricname}")
    get_tlambda = lambda metrics, lamb: [i[0] for i in metrics.items() if i[1][metricname] >= prim_lambda]
    get_tlambda2 = lambda metrics, primlamb, seclamb: [i[0] for i in metrics.items() if i[1][metricname] >= sec_lambda and i[1][metricname] < prim_lambda]
    candidates = get_tlambda(metrics, prim_lambda)
    salient_directions = [max(metrics.items(), key=lambda x: x[1][metricname])[0],]
    n_terms = min(len(candidates), 2*len(decision_planes[salient_directions[0]].coef)) #from [DESC15]
    comparer = Comparer(decision_planes, vec_cos)
    for nterm in tqdm(range(1, n_terms), desc="Merging Salient Directions"):
        cands = set(candidates)-set(salient_directions)
        compares = {cand: max(comparer(cand, compareto) for compareto in salient_directions) for cand in cands}
        salient_directions.append(min(compares.items(), key=lambda x:x[1])[0])
    print(f"Found {len(salient_directions)} salient directions: {salient_directions}")
    compare_vecs = [decision_planes[term].normal for term in salient_directions]
    clusters = {term: [] for term in salient_directions}
    #TODO optionally instead do the cluster-assignment with k-means!
    for term in tqdm(get_tlambda2(metrics, prim_lambda, sec_lambda), desc="Associating Clusters"):
        # "we then associate with each term d_i a Cluster C_i containing all terms from T^{0.1} which are more similar to d_i than to any of the
        # other directions d_j." TODO: experiment with thresholds, if it's extremely unsimilar to ALL just effing discard it!
        clusters[salient_directions[np.argmin([vec_cos(decision_planes[term].normal, vec2) for vec2 in compare_vecs])]].append(term)
    cluster_directions = {key: np.mean(np.array([decision_planes[term].normal for term in [key]+vals]), axis=0) for key, vals in clusters.items()}
    # TODO maybe have a smart weighting function that takes into account the kappa-score of the term and/or the closeness to the original clustercenter
    return clusters, cluster_directions

def create_candidate_svm(embedding, term, quants, plot_svm=False, descriptions=None, quant_name=None, pgbar=None, **kwargs):
    bin_labels = np.array(quants, dtype=bool) # Ensure that regardless of quant_measure this is correct binary classification labels
    # (tmp := len(quants)/(2*np.bincount(bin_labels)))[0]/tmp[1] is roughly equal to bin_labels.mean() so balancing is good
    # see https://stackoverflow.com/q/33843981/5122790, https://stackoverflow.com/q/35076586/5122790
    # svm = sklearn.svm.SVC(kernel="linear", class_weight="balanced")  #slow as fuck
    # svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced") #squared-hinge instead of hinge (but fastest!)
    svm = sklearn.svm.LinearSVC(class_weight="balanced", loss="hinge", max_iter=20000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svm.fit(embedding, bin_labels)
        no_converge = False
        if w:
            assert len(w) == 1
            assert issubclass(w[0].category, sklearn.exceptions.ConvergenceWarning)
            no_converge = True
    svm_results = svm.decision_function(embedding)
    tn, fp, fn, tp = confusion_matrix(bin_labels, [i > 0 for i in svm_results]).ravel()
    precision = tp / (tp + fp); recall = tp / (tp + fn); accuracy = (tp + tn) / len(quants)
    f_one = 2*(precision*recall)/(precision+recall)
    res = {"accuracy": accuracy, "precision": precision, "recall": recall, "f_one": f_one, "did_converge": not no_converge}
    # print(f"accuracy: {accuracy:.2f} | precision: {precision:.2f} | recall: {recall:.2f}")
    #now, in [DESC15:4.2.1], they compare the "ranking induced by \vec{v_t} with the number of times the term occurs in the entity's documents" with Cohen's Kappa.

    #see notebooks/proof_of_concept/get_svm_decisionboundary.ipynb#Checking-projection-methods-&-distance-measures-from-point-to-projection for the ranking
    decision_plane = NDPlane(svm.coef_[0], svm.intercept_[0])  #don't even need the plane class here
    dist = lambda x, plane: np.dot(plane.normal, x) + plane.intercept
    distances = [dist(point, decision_plane) for point in embedding]
    #sanity check: do most of the points with label=0 have the same sign `np.count_nonzero(np.sign(np.array(distances)[bin_labels])+1)`
    # quant_ranking = np.zeros(quants.shape); quant_ranking[np.where(quants > 0)] = np.argsort(quants[quants > 0])
    #TODO cohen's kappa hat nen sample_weight parameter!! DESC15 write they select Kappa "due to its tolerance to class imbalance." -> Does that mean I have to set the weight?!
    res["kappa_rank2rank_dense"]  = cohen_kappa_score(rankdata(quants, method="dense"), rankdata(distances, method="dense")) #if there are 14.900 zeros, the next is a 1
    res["kappa_rank2rank_min"] = cohen_kappa_score(rankdata(quants, method="min"), rankdata(distances, method="dense")) #if there are 14.900 zeros, the next one is a 14.901
    res["kappa_bin2bin"]    = cohen_kappa_score(bin_labels, [i > 0 for i in distances])
    res["kappa_digitized"]  = cohen_kappa_score(np.digitize(quants, np.histogram_bin_edges(quants)[1:]), np.digitize(distances, np.histogram_bin_edges(distances)[1:]))
    nonzero_indices = np.where(np.array(quants) > 0)[0]
    q2, d2 = np.array(quants)[nonzero_indices], np.array(distances)[nonzero_indices]
    with warnings.catch_warnings(): #TODO get rid of what cuases the nans here!!!
        warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
        if quant_name == "count":  # in DESC15 they write "measure the correlation between the ranking induced by \vec{vt} and the number of times t appears in the documents associated with each entity", so maybe compare ranking to count?!
            res["kappa_count2rank"] = cohen_kappa_score(quants, rankdata(distances, method="dense"))
            res["kappa_count2rank_onlypos"] = cohen_kappa_score(q2, rankdata(d2, method="dense"))
        res["kappa_rank2rank_onlypos_dense"] = cohen_kappa_score(rankdata(q2, method="dense"), rankdata(d2, method="dense"))
        res["kappa_rank2rank_onlypos_min"] = cohen_kappa_score(rankdata(q2, method="min"), rankdata(d2, method="min"))
        res["kappa_digitized_onlypos_1"] = cohen_kappa_score(np.digitize(q2, np.histogram_bin_edges(quants)[1:]), np.digitize(d2, np.histogram_bin_edges(distances)[1:]))
        #one ^ has as histogram-bins what it would be for ALL data, two only for the nonzero-ones
        res["kappa_digitized_onlypos_2"] = cohen_kappa_score(np.digitize(q2, np.histogram_bin_edges(q2)[1:]), np.digitize(d2, np.histogram_bin_edges(d2)[1:]))
    if plot_svm and descriptions is not None:
        display_svm(embedding, np.array(bin_labels, dtype=int), svm, term=term, descriptions=descriptions, name=term+" "+(", ".join(f"{k}: {round(v, 3)}" for k, v in res.items())), **kwargs)
    if pgbar is not None:
        pgbar.update(1)
    return res, decision_plane, term



def display_svm(X, y, svm, term=None, name=None, descriptions=None, highlight=None):
    assert X.shape[1] == 3
    decision_plane = ThreeDPlane(svm.coef_[0], svm.intercept_[0])
    occurences = [descriptions._descriptions[i].count_phrase(term) for i in range(len(X))]
    percentile = lambda percentage: np.percentile(np.array([i for i in occurences if i]), percentage)
    if descriptions._descriptions[0].text is not None:
        extras = [{"Name": descriptions._descriptions[i].title, "Occurences": occurences[i], "extra": {"Description": shorten(descriptions._descriptions[i].text, 200)}} for i in range(len(X))]
    else:
        extras = [{"Name": descriptions._descriptions[i].title, "Occurences": occurences[i], "extra": {"BoW": ", ".join([f"{k}: {v}" for k, v in sorted(descriptions._descriptions[i].bow().items(), key=lambda x:x[1], reverse=True)[:10]])}} for i in range(len(X))]
    highlight_inds = [n for n, i in enumerate(descriptions._descriptions) if i.title in highlight] if highlight else []
    with ThreeDFigure(name=name) as fig:
        fig.add_markers(X[np.where(np.logical_not(y))], color="blue", size=2, custom_data=[extras[i] for i in np.where(np.logical_not(y))[0]], linelen_right=50, name="negative samples")
        fig.add_markers(X[np.where(y)], color="red", size=[9 if occurences[i] > percentile(70) else 4 for i in np.where(y)[0]], custom_data=[extras[i] for i in np.where(y)[0]], linelen_right=50, name="positive samples")
        if highlight_inds:
            highlight_mask = np.array([i in highlight_inds for i in range(len(y))], dtype=int)
            fig.add_markers(X[np.where(highlight_mask)], color="green", size=9, custom_data=[extras[i] for i in np.where(highlight_mask)[0]], linelen_right=50, name="highlighted")
        fig.add_surface(decision_plane, X, y, color="gray")  # decision hyperplane
        fig.add_line(X.mean(axis=0)-decision_plane.normal*2, X.mean(axis=0)+decision_plane.normal*2, width=2, name="Orthogonal")  # orthogonal of decision hyperplane through mean of points
        fig.add_markers([0, 0, 0], size=3, name="Coordinate Center")  # coordinate center
        # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        fig.show()