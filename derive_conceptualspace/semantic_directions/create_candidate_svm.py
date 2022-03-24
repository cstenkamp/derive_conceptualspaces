from contextlib import nullcontext
from itertools import repeat
import warnings
from textwrap import shorten
import random
import time
from collections import Counter

from tqdm import tqdm
import sklearn.svm
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import rankdata
import pandas as pd

from derive_conceptualspace.settings import get_setting, IS_INTERACTIVE, get_ncpu
from derive_conceptualspace.util.base_changer import NDPlane, ThreeDPlane
from derive_conceptualspace.util.dtm_object import DocTermMatrix
from derive_conceptualspace.util.interruptible_funcs import Interruptible, SkipContext
from derive_conceptualspace.util.threedfigure import ThreeDFigure
from derive_conceptualspace.util.threadpool import ThreadPool
from misc_util.pretty_print import pretty_print as print


norm = lambda vec: vec/np.linalg.norm(vec)
vec_cos = lambda v1, v2: np.arccos(np.clip(np.dot(norm(v1), norm(v2)), -1.0, 1.0))  #https://stackoverflow.com/a/13849249/5122790
#`vec_cos((1, 0, 0), (0, 1, 0)) == vec_cos((1, 0, 0), (0, 1, 1))` --> that's correct, they ARE 90° if they are (1,0,..) and (0,1,..), never mind other coordinates

flatten = lambda l: [item for sublist in l for item in sublist]
unique = lambda iterable: list({i:None for i in iterable}.keys())


def cohen_kappa(y_test, y_pred, **kwargs):
    """see https://github.com/scikit-learn/scikit-learn/issues/9624#issuecomment-1004342697"""
    if len(set(y_test).union(y_pred)) == 1:
        return 1
    return cohen_kappa_score(y_test, y_pred, **kwargs)


def create_candidate_svms(dcm, embedding, descriptions, verbose, continue_from=None):
    #TODO I am still not sure about if I am calculating with vectors somewhere where when I should be working with points
    decision_planes = {}
    metrics = {}
    terms = list(dcm.all_terms.values())
    metainf = {}
    if get_setting("DEBUG"):
        maxlen = min(len(terms), len(embedding.embedding_), get_setting("DEBUG_N_ITEMS"), len(dcm.dtm))
        working_inds = [nterm for nterm, term in enumerate(terms[:maxlen]) if np.array(dcm.term_quants(term)[:maxlen], dtype=bool).std()] #those with >1 class
        term_inds = unique(flatten([j[0] for j in dcm.dtm[i]] for i in working_inds))
        terms = [dcm.all_terms[i] for i in term_inds]
        embedding.embedding_ = embedding.embedding_[working_inds]
        ind_translator = {v: k for k, v in enumerate(term_inds)}
        dcm = DocTermMatrix([[[ind_translator[j[0]],j[1]] for j in dcm.dtm[i]] for i in working_inds],
                            {ind_translator[i]: dcm.all_terms[i] for i in term_inds}, dcm.quant_name)
        print(f"Debug-Mode: Running for {len(working_inds)} Items and {len(terms)} Terms.")
        # warnings.warn("PRECOMMIT there's stuff here!")
        # assert all(i in terms for i in ['nature', 'ceiling', 'engine', 'athlete', 'seafood', 'shadows', 'skyscrapers', 'b737', 'monument', 'baby', 'sign', 'marine', 'iowa', 'field', 'buy', 'military', 'lounge', 'factory', 'road', 'education', '13thcentury', 'people', 'wait', 'travel', 'tunnel', 'treno', 'wings', 'hot', 'background', 'vintage', 'farmhouse', 'technology', 'building', 'horror', 'realestate', 'crane', 'slipway', 'ruin', 'national', 'morze'])
        # terms = ['nature', 'ceiling', 'engine', 'athlete', 'seafood', 'shadows', 'skyscrapers', 'b737', 'monument', 'baby', 'sign', 'marine', 'iowa', 'field', 'buy', 'military', 'lounge', 'factory', 'road', 'education', '13thcentury', 'people', 'wait', 'travel', 'tunnel', 'treno', 'wings', 'hot', 'background', 'vintage', 'farmhouse', 'technology', 'building', 'horror', 'realestate', 'crane', 'slipway', 'ruin', 'national', 'morze']
        # assert len([i for i in descriptions._descriptions if 'nature' in i]) == len([i for i in dcm.term_quants('nature') if i > 0])
        # print(f"Running only for the terms {terms}")
    else:
        assert all(len([i for i in descriptions._descriptions if term in i]) == len([i for i in dcm.term_quants(term) if i > 0]) for term in random.sample(terms, 5))
    if get_setting("DO_SANITYCHECKS"):
        assert all(dcm.term_quants(terms[i]) == list(dcm.as_csr()[i,:].toarray().squeeze()) for i in random.sample(range(len(terms)), 5))

    quants_s = dcm.as_csr().toarray().tolist()  # [dcm.term_quants(term) for term in tqdm(terms, desc="Counting Terms")]
    ncpu = get_ncpu(ram_per_core=10) #TODO: make ram_per_core dependent on dataset-size
    if ncpu == 1:  #TODO Interruptible: for ncpu==1, I'm adding direct key-value-pairs, in the ncpu>1 version I'm appending to a list -> they are incompatible!
        with Interruptible(zip(terms, quants_s), ([], decision_planes, metrics), metainf, continue_from=continue_from, pgbar="Creating Candidate SVMs [1 proc]", total=len(terms), name="SVMs") as iter:
            for term, quants in iter: #in tqdm(zip(terms, quants_s), desc="Creating Candidate SVMs", total=len(terms))
                cand_mets, decision_plane, term = create_candidate_svm(embedding.embedding_, term, quants, classifier=get_setting("CLASSIFIER"), descriptions=descriptions, quant_name=dcm.quant_name)
                metrics[term] = cand_mets
                decision_planes[term] = decision_plane
    else:
        print(f"Starting Multiprocessed with {ncpu} CPUs")
        with Interruptible(zip(terms, quants_s), [None, [], None], metainf, continue_from=continue_from, contains_mp=True, name="SVMs", total=len(quants_s)) as iter:
            with tqdm(total=iter.n_elems, desc=f"Creating Candidate SVMs [{ncpu} procs]") as pgbar, ThreadPool(ncpu, comqu=iter.comqu) as p:
                res, interrupted = p.starmap(create_candidate_svm, zip(repeat(embedding.embedding_, iter.n_elems), repeat("next_0"), repeat("next_1"), repeat(get_setting("CLASSIFIER")), repeat(False), repeat(None), repeat(dcm.quant_name), repeat(pgbar)), draw_from=iter.iterable)
            _, res, _ = iter.notify([None, res, None], exception=interrupted)
            if interrupted is not False:
                return quants_s, res, None, metainf
        for cand_mets, decision_plane, term in res:
            metrics[term] = cand_mets
            decision_planes[term] = decision_plane
        assert set(terms) == set(metrics.keys())
    if (didnt_converge := len([1 for i in metrics.values() if i and not i["did_converge"]])):
        warnings.warn(f"{didnt_converge} of the {len(metrics)} SVMs did not converge!", sklearn.exceptions.ConvergenceWarning)
    if verbose:
        df = pd.DataFrame(metrics).T
        df.columns = df.columns.str.replace("kappa", "k").str.replace("rank2rank", "r2r").str.replace("bin2bin", "b2b").str.replace("f_one", "f1").str.replace("digitized", "dig")
        for metricname in df.columns:
            print(f"\nAverage *r*{metricname}*r*: {df[metricname].mean():.5f}")
            with pd.option_context('display.max_rows', 11, 'display.max_columns', 20, 'display.expand_frame_repr', False, 'display.max_colwidth', 20, 'display.float_format', '{:.4f}'.format):
                print(str(df.sort_values(by=metricname, ascending=False)[:10]).replace(metricname, f"*r*{metricname}*r*"))
        if embedding.embedding_.shape[1] == 3 and IS_INTERACTIVE:
            best_elem = max(metrics.items(), key=lambda x:(x[1] or {}).get("f_one",0))
            create_candidate_svm(embedding.embedding_, best_elem[0], dcm.term_quants(best_elem[0]), classifier=get_setting("CLASSIFIER"), quant_name=dcm.quant_name, plot_svm=True, descriptions=descriptions)
            while (another := input("Another one to display: ").strip()) != "":
                if "," in another:
                    highlight = [i.strip() for i in another.split(",")[1:]]
                    another = another.split(",")[0].strip()
                else:
                    highlight = []
                create_candidate_svm(embedding.embedding_, another, dcm.term_quants(another), classifier=get_setting("CLASSIFIER"), quant_name=dcm.quant_name, plot_svm=True, descriptions=descriptions, highlight=highlight)
    return quants_s, decision_planes, metrics, metainf


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


def select_salient_terms(metrics, decision_planes, dcm, embedding, prim_lambda, sec_lambda, metricname, verbose=False):
    #TODO waitwaitwait. Am I 100% sure that the intercepts of the decision_planes are irrelevant?!
    #TODO what about those with high negative kappa? Einfach abs-wert nehmen und consideren (AUCH SCHON IM SCHRITT VORHER IF SO)
    print(f"Calculated Metrics: {list(list(metrics.values())[0].keys())}")
    print(f"Lambda1: {prim_lambda}, Lambda2: {sec_lambda}, compareto-metric: {metricname}")
    metrics = sorted(list({k: v[metricname] for k, v in metrics.items()}.items()), key=lambda x:x[1], reverse=True)
    get_tlambda = lambda metrics, lamb: [i[0] for i in metrics if i[1] >= prim_lambda]
    get_tlambda2 = lambda metrics, lamb1objs, seclamb: [i[0] for i in metrics if i[1] >= sec_lambda and i[0] not in lamb1objs]
    candidates = get_tlambda(metrics, prim_lambda)
    salient_directions = [metrics[0][0],]
    n_terms = min(len(candidates), get_setting("NDIMS_NCANDS_FACTOR")*len(decision_planes[salient_directions[0]].coef)) #2 in [DESC15]
    if get_setting("DEBUG"): n_terms = min(n_terms, 15)
    comparer = Comparer(decision_planes, vec_cos)
    #DESC15: "as the ith term, we select the term t minimising max_{j<i}cos(v_t_j, v_t) - In other words, we repeatedly select the term which is least similar to the terms that have already been selected"
    for nterm in tqdm(range(1, n_terms), desc="Finding Salient Directions"):
        cands = set(candidates)-set(salient_directions)
        compares = {cand: min(comparer(cand, compareto) for compareto in salient_directions) for cand in cands}
        #vec_cos(decision_planes[next(iter(cands))].normal, decision_planes[salient_directions[0]].normal)
        salient_directions.append(max(compares.items(), key=lambda x:x[1])[0])
    print(f"Found {len(salient_directions)} salient directions: {', '.join(salient_directions)}")
    compare_vecs = [decision_planes[term].normal for term in salient_directions]
    clusters = {term: [] for term in salient_directions}
    #TODO optionally instead do the cluster-assignment with k-means!
    nongreats = get_tlambda2(metrics, salient_directions, sec_lambda)
    if get_setting("DEBUG"): nongreats = nongreats[:2000]
    for term in tqdm(nongreats, desc="Associating the rest to Clusters"):
        # "we then associate with each term d_i a Cluster C_i containing all terms from T^{0.1} which are more similar to d_i than to any of the
        # other directions d_j." TODO: experiment with thresholds, if it's extremely unsimilar to ALL just effing discard it!
        clusters[salient_directions[np.argmin([vec_cos(decision_planes[term].normal, vec2) for vec2 in compare_vecs])]].append(term)
    # TODO maybe have a smart weighting function that takes into account the kappa-score of the term and/or the closeness to the original clustercenter (to threshold which cluster they are added to)

    #TODO an option here to either take mean, or only main-one, or smartly-weighted (I think DESC15 did only main-one)
    if get_setting("CLUSTER_DIRECTION_ALGO") == "mean":
        cluster_directions = join_clusters_average(clusters, decision_planes)
    elif get_setting("CLUSTER_DIRECTION_ALGO") == "main":
        cluster_directions = {term: decision_planes[term] for term in clusters.keys()}
    elif get_setting("CLUSTER_DIRECTION_ALGO") == "reclassify":
        cluster_directions = join_clusters_reclassify(clusters, dcm, embedding, verbose=verbose)
    else:
        raise NotImplementedError("TODO: weighted and others")
        #missing: weighted-by-kappa-averaged, weighted-by-distance-to-center-averaged (cosine, cosine+coef)
    #regarding mean-algorithm: taking the mean of the respective orthogonals seems reasonable, it's the mean direction. However we also care for the actual position of the
    # hyperplane (to get the actual ranking-wrt-this-feature), which is specified by orthogonal+intercept... and simply averaging the intercepts of it's clustercomponents seems really stupid.
    #  that however gives us another way to weight which-candidates-may-cluster: the closer the orthogonals (cosine-dist) AND the closer their intercepts, the more we want to have them in a cluster.

    return clusters, cluster_directions
    #TODO in this step, there are SO MANY improvements to be made.
    # * I am ordering by kappa and selecting from there on - because "isawyoufirst" is very close to "nature", "nature" will never be picked out as term
    #    -> either find a way to select more... informative?? terms from the start, or find a way to, once the corresponding clusters are found, select a good one of that as representative
    #        -> use LSI for that (add pseudodocs with only one term and that term as name, and then let all DOCUMENTNAMES be candidates (change in EARLIER STEP!))
    #        -> do it like the "enhancing from wikipedia" paper and after the whole clustering, consider the documentnames that fit best as clusterlabels
    # * The fact that for the clustering, all of the T^0.1 ones are added to the "most similar one", I would definititely threshold that, those that are far from are not considered
    # * Like Ager2018 or Alshaikh2020, I would LIKE to get uninformative clusters, such that I can throw them out.
    # * Well, the hierachical way of Alshaikh2020 seemed promising, didn't it?!
    # * Calculate the direction of the cluster anew,
    # * 	a) by the (weighted-by-kappa-or-closeness-to-center) averaged direction of it's members,
    # *     b) by creating a new SVM with "any(or-at-least-x)-of-the-cluster-terms-occur"

################### methods to join clusters ###################

def join_clusters_average(clusters, decision_planes):
    res = {}
    for key, vals in clusters.items():
        mean_coef = np.mean(np.array([decision_planes[term].coef for term in [key]+vals]), axis=0)
        mean_interc = np.mean(np.array([decision_planes[term].intercept for term in [key]+vals]))
        res[key] = NDPlane(mean_coef, mean_interc)
    return res


def join_clusters_reclassify(clusters, dcm, embedding, verbose=False):
    if hasattr(embedding, "embedding_"): embedding = embedding.embedding_
    all_cand_mets = {}
    cluster_directions = {}
    for k, v in tqdm(clusters.items(), desc="Reclassifying Clusters"):
        embed = embedding
        dtm = DocTermMatrix.submat_forterms(dcm, [k] + v)
        combined_quants = dtm.as_csr().toarray().sum(axis=0)
        if any(i < get_setting("CANDIDATE_MIN_TERM_COUNT") or i > dtm.n_docs-get_setting("CANDIDATE_MIN_TERM_COUNT") for i in Counter(np.array(combined_quants, dtype=bool)).values()):
            #TODO have an option for doing this GENERALLY for the SVMs (and plot in 3D)
            c0_inds = np.where(combined_quants <= np.percentile(combined_quants, 30))[0]
            c1_inds = np.where(combined_quants >= np.percentile(combined_quants, 70))[0]
            used_inds = sorted(list(set(c0_inds)|set(c1_inds)))
            embed = embedding[used_inds]
            if verbose:
                print(f"For cluster {k}, the distribution is {dict(Counter(np.array(combined_quants, dtype=bool)))}, so we'll take the most distinct {get_setting('MOST_DISTINCT_PERCENT')}% ({len(c0_inds)} entities per class)")
            combined_quants = [combined_quants[i] if i in c1_inds else 0 for i in used_inds]
        cand_mets, decision_plane, _ = create_candidate_svm(embed, f"cluster:{k}", combined_quants, get_setting("CLASSIFIER"), quant_name=dtm.quant_name)
        all_cand_mets[k] = cand_mets
        cluster_directions[k] = decision_plane
    if verbose:
        print(f"Scores for {get_setting('CLASSIFIER_SUCCMETRIC')} per cluster:", ", ".join(f"{k}: {v[get_setting('CLASSIFIER_SUCCMETRIC')]:.2f}" for k, v in all_cand_mets.items()))
    return cluster_directions


################### methods to join clusters END ###############



def create_candidate_svm(embedding, term, quants, classifier, plot_svm=False, descriptions=None, quant_name=None, pgbar=None, **kwargs):
    #!! term is only used for visualization, and ist must stay that way for CLUSTER_DIRECTION_ALGO = "reclassify" !
    bin_labels = np.array(quants, dtype=bool) # Ensure that regardless of quant_measure this is correct binary classification labels
    # (tmp := len(quants)/(2*np.bincount(bin_labels)))[0]/tmp[1] is roughly equal to bin_labels.mean() so balancing is good
    if classifier == "SVM":
        svm = sklearn.svm.LinearSVC(class_weight="balanced", loss="hinge", max_iter=20000)
    elif classifier == "SVM_square":
        svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced") #squared-hinge instead of hinge (but fastest!)
    elif classifier == "SVM2":
        warnings.warn("Using an SVM Implementation that's slower for this kind of data!")
        svm = sklearn.svm.SVC(kernel="linear", class_weight="balanced", decision_function_shape="ovo")  #slower than LinearSVC, don't use!
        # see https://stackoverflow.com/q/33843981/5122790, https://stackoverflow.com/q/35076586/5122790
    else:
        raise NotImplementedError(f"Demanded classifier {classifier} not implemented!")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svm.fit(embedding, bin_labels)
        if w: assert issubclass(w[0].category, (sklearn.exceptions.ConvergenceWarning, DeprecationWarning))
        no_converge = (bool(w) and issubclass(w[0].category, sklearn.exceptions.ConvergenceWarning))
    tn, fp, fn, tp = confusion_matrix(bin_labels, svm.predict(embedding)).ravel()
    res = {"accuracy": (tp + tn) / len(quants), "precision": tp / (tp + fp), "recall": tp / (tp + fn), "did_converge": not no_converge}
    res["f_one"] = 2 * (res["precision"] * res["recall"]) / (res["precision"] + res["recall"])
    #now, in [DESC15:4.2.1], they compare the "ranking induced by \vec{v_t} with the number of times the term occurs in the entity's documents" with Cohen's Kappa.

    #see notebooks/proof_of_concept/get_svm_decisionboundary.ipynb#Checking-projection-methods-&-distance-measures-from-point-to-projection for the ranking
    decision_plane = NDPlane(svm.coef_[0], svm.intercept_[0])  #don't even need the plane class here
    dist = lambda x, plane: np.dot(plane.normal, x) + plane.intercept
    distances = [dist(point, decision_plane) for point in embedding]
    assert np.allclose(distances, svm.decision_function(embedding)) #see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.decision_function, https://stats.stackexchange.com/a/14881
    distances /= np.linalg.norm(svm.coef_[0]) #TODO: add the links and this normalification to the distances-notebook
    #sanity check: do most of the points with label=0 have the same sign `np.count_nonzero(np.sign(np.array(distances)[bin_labels])+1)
    # bin_labels, np.array((np.sign(np.array(distances))+1)/2, dtype=bool)
    # quant_ranking = np.zeros(quants.shape); quant_ranking[np.where(quants > 0)] = np.argsort(quants[quants > 0])
    #TODO cohen's kappa hat nen sample_weight parameter!! DESC15 write they select Kappa "due to its tolerance to class imbalance." -> Does that mean I have to set the weight?!
    kappa_weights = get_setting("KAPPA_WEIGHTS") if get_setting("KAPPA_WEIGHTS") != "None" else None
    res["kappa_rank2rank_dense"]  = cohen_kappa(rankdata(quants, method="dense"), rankdata(distances, method="dense"), weights=kappa_weights) #if there are 14.900 zeros, the next is a 1
    res["kappa_rank2rank_min"] = cohen_kappa(rankdata(quants, method="min"), rankdata(distances, method="dense"), weights=kappa_weights) #if there are 14.900 zeros, the next one is a 14.901
    res["kappa_bin2bin"]    = cohen_kappa(bin_labels, [i > 0 for i in distances], weights=kappa_weights)
    res["kappa_digitized"]  = cohen_kappa(np.digitize(quants, np.histogram_bin_edges(quants)[1:]), np.digitize(distances, np.histogram_bin_edges(distances)[1:]), weights=kappa_weights)
    nonzero_indices = np.where(np.array(quants) > 0)[0]
    q2, d2 = np.array(quants)[nonzero_indices], np.array(distances)[nonzero_indices]
    with nullcontext(): #warnings.catch_warnings(): #TODO get rid of what cuases the nans here!!!
        # warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
        if quant_name == "count":  # in DESC15 they write "measure the correlation between the ranking induced by \vec{vt} and the number of times t appears in the documents associated with each entity", so maybe compare ranking to count?!
            # res["kappa_count2rank"] = cohen_kappa(quants, rankdata(distances, method="dense"), weights=kappa_weights)
            res["kappa_count2rank_onlypos"] = cohen_kappa(q2, rankdata(d2, method="dense"), weights=kappa_weights)
        res["kappa_rank2rank_onlypos_dense"] = cohen_kappa(rankdata(q2, method="dense"), rankdata(d2, method="dense"), weights=kappa_weights)
        res["kappa_rank2rank_onlypos_min"] = cohen_kappa(rankdata(q2, method="min"), rankdata(d2, method="min"), weights=kappa_weights)
        res["kappa_rank2rank_onlypos_max"] = cohen_kappa(rankdata(q2, method="max"), rankdata(d2, method="max"), weights=kappa_weights)
        # res["kappa_digitized_onlypos_1"] = cohen_kappa(np.digitize(q2, np.histogram_bin_edges(quants)[1:]), np.digitize(d2, np.histogram_bin_edges(distances)[1:]), weights=kappa_weights)
        #one ^ has as histogram-bins what it would be for ALL data, two only for the nonzero-ones
        res["kappa_digitized_onlypos_2"] = cohen_kappa(np.digitize(q2, np.histogram_bin_edges(q2)[1:]), np.digitize(d2, np.histogram_bin_edges(d2)[1:]), weights=kappa_weights)
    if plot_svm and descriptions is not None:
        display_svm(embedding, np.array(bin_labels, dtype=int), svm, term=term, descriptions=descriptions, name=term+" "+(", ".join(f"{k}: {round(v, 3)}" for k, v in res.items())), quants=quants, distances=distances, **kwargs)
    if pgbar is not None:
        pgbar.update(1)
    return res, decision_plane, term



def display_svm(X, y, svm, term=None, name=None, descriptions=None, quants=None, distances=None, highlight=None, stretch_fact=2, **kwargs):
    assert X.shape[1] == 3
    decision_plane = ThreeDPlane(svm.coef_[0], svm.intercept_[0])
    occurences = [descriptions._descriptions[i].count_phrase(term) for i in range(len(X))]
    percentile = lambda percentage: np.percentile(np.array([i for i in occurences if i]), percentage)
    if descriptions._descriptions[0].text is not None:
        extras = [{**{"Name": descriptions._descriptions[i].title, "Occurences": occurences[i],
                      "extra": {"Description": shorten(descriptions._descriptions[i].text, 200)}},
                   **({"Quants": quants[i]} if quants is not None else {}),
                   **({"Distance": distances[i]} if distances is not None else {})}
                  for i in range(len(X))]
    else:
        extras = [{**{"Name": descriptions._descriptions[i].title, "Quants": quants[i], "Occurences": occurences[i],
                      "extra": {"BoW": ", ".join([f"{k}: {v}" for k, v in sorted(descriptions._descriptions[i].bow().items(), key=lambda x:x[1], reverse=True)[:10]])}},
                   **({"Quants": quants[i]} if quants is not None else {}),
                   **({"Distance": distances[i]} if distances is not None else {})}
                  for i in range(len(X))]
    highlight_inds = [n for n, i in enumerate(descriptions._descriptions) if i.title in highlight] if highlight else []
    with ThreeDFigure(name=name, **kwargs) as fig:
        fig.add_markers(X[np.where(np.logical_not(y))], color="blue", size=2, custom_data=[extras[i] for i in np.where(np.logical_not(y))[0]], linelen_right=50, name="negative samples")
        fig.add_markers(X[np.where(y)], color="red", size=[9 if occurences[i] > percentile(70) else 4 for i in np.where(y)[0]], custom_data=[extras[i] for i in np.where(y)[0]], linelen_right=50, name="positive samples")
        if highlight_inds:
            highlight_mask = np.array([i in highlight_inds for i in range(len(y))], dtype=int)
            fig.add_markers(X[np.where(highlight_mask)], color="green", size=9, custom_data=[extras[i] for i in np.where(highlight_mask)[0]], linelen_right=50, name="highlighted")
        fig.add_surface(decision_plane, X, y, color="gray")  # decision hyperplane
        fig.add_line(X.mean(axis=0)-decision_plane.normal*stretch_fact, X.mean(axis=0)+decision_plane.normal*stretch_fact, width=2, name="Orthogonal")  # orthogonal of decision hyperplane through mean of points
        fig.add_markers([0, 0, 0], size=3, name="Coordinate Center")  # coordinate center
        # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        fig.show()