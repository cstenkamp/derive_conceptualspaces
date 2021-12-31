from textwrap import shorten

from tqdm import tqdm
import sklearn.svm
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.base_changer import NDPlane, ThreeDPlane
from derive_conceptualspace.util.threedfigure import ThreeDFigure

norm = lambda vec: vec/np.linalg.norm(vec)
vec_cos = lambda v1, v2: np.arccos(np.clip(np.dot(norm(v1), norm(v2)), -1.0, 1.0))  #https://stackoverflow.com/a/13849249/5122790


def create_candidate_svms(dcm, mds, pp_descriptions, prim_lambda, sec_lambda, verbose):
    _, descriptions = pp_descriptions.values()
    metrics = {}
    decision_planes = {}
    for term, exist_indices in tqdm(dcm.term_existinds(use_index=False).items()):
        if not get_setting("DEBUG"):
            assert len(exist_indices) >= get_setting("CANDIDATE_MIN_TERM_COUNT") #TODO this is relevant-metainf!!
        cand_mets, decision_plane = create_candidate_svm(mds, term, exist_indices, descriptions)
        metrics[term] = cand_mets
        decision_planes[term] = decision_plane
    if verbose:
        for metricname in list(metrics.values())[0].keys():
            print(f"\nAverage {metricname}: {sum(i[metricname] for i in metrics.values())/len(metrics)*100:.3f}%")
            sorted_by = sorted([[k,v] for k,v in metrics.items()], key=lambda x:x[1][metricname], reverse=True)[:10]
            strings = [f"{term.ljust(max(len(i[0]) for i in sorted_by))}: "+
                       ", ".join(f"{k}: {v*100:5.2f}%" for k,v in metrics.items() if k != "kappa")+
                       f", kappa: {metrics['kappa']:.4f}"+f", samples: {len(dcm.term_existinds(use_index=False)[term])}"
                       for term, metrics in sorted_by[:20]]
            print("  "+"\n  ".join(strings))
    sorted_kappa = [(i[0], i[1]["kappa"]) for i in sorted(metrics.items(), key=lambda x:x[1]["kappa"], reverse=True)]
    if verbose and mds.embedding_.shape[1] == 3:
        create_candidate_svm(mds, sorted_kappa[0][0], dcm.term_existinds(use_index=False)[sorted_kappa[0][0]], descriptions, plot_svm=True)
        while (another := input("Another one to display: ").strip()) != "" or another not in dcm.term_existinds(use_index=False):
            create_candidate_svm(mds, another, dcm.term_existinds(use_index=False)[another], descriptions, plot_svm=True)
    clusters, cluster_directions = select_salient_terms(sorted_kappa, decision_planes, prim_lambda, sec_lambda)
    return clusters, cluster_directions, decision_planes, metrics


def select_salient_terms(sorted_kappa, decision_planes, prim_lambda, sec_lambda):
    #TODO waitwaitwait. Am I 100% sure that the intercepts of the decision_planes are irrelevant?!
    get_tlambda = lambda sorted_kappa, lamb: [i[0] for i in sorted_kappa if i[1] > lamb]
    get_tlambda2 = lambda sorted_kappa, primlamb, seclamb: list(set(get_tlambda(sorted_kappa, seclamb))-set(get_tlambda(sorted_kappa, primlamb)))
    candidates = get_tlambda(sorted_kappa, prim_lambda)
    salient_directions = [sorted_kappa[0][0],]
    for nterm in tqdm(range(1, len(candidates))):
        compare_vecs = [decision_planes[term].normal for term in salient_directions]
        cands = set(candidates)-set(salient_directions)
        # TODO mit cachen lässt sich hier EXTREM viel beschleunigen
        compares = {cand: max([vec_cos(decision_planes[cand].normal, vec2) for vec2 in compare_vecs]) for cand in cands}
        salient_directions.append(min(compares.items(), key=lambda x:x[1])[0])
        if len(salient_directions) >= 2*len(decision_planes[salient_directions[0]].coef):
            break #from [DESC15]
    print(f"Found {len(salient_directions)} salient directions: {salient_directions}")
    compare_vecs = [decision_planes[term].normal for term in salient_directions]
    clusters = {term: [] for term in salient_directions}
    #TODO instead do the cluster-assignment with k-means!
    for term in tqdm(get_tlambda2(sorted_kappa, prim_lambda, sec_lambda)):
        # "we then associate with each term d_i a Cluster C_i containing all terms from T^{0.1} which are more similar to d_i than to any of the
        # other directions d_j." TODO: experiment with thresholds, if it's extremely unsimilar to ALL just effing discard it!
        clusters[salient_directions[np.argmin([vec_cos(decision_planes[term].normal, vec2) for vec2 in compare_vecs])]].append(term)
    cluster_directions = {key: np.mean(np.array([decision_planes[term].normal for term in [key]+vals]), axis=0) for key, vals in clusters.items()}
    # TODO maybe have a smart weighting function that takes into account the kappa-score of the term and/or the closeness to the original clustercenter
    return clusters, cluster_directions


def create_candidate_svm(embedding, term, exist_indices, descriptions, plot_svm=False):
    #TODO [DESC15]: "we adapted the costs of the training instances to deal with class imbalance (using the ratio between entities with/without the term as cost)"
    # TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
    labels = [False] * embedding.embedding_.shape[0]
    for i in exist_indices:
        labels[i] = True
    svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
    svm.fit(embedding.embedding_, np.array(labels, dtype=int))
    svm_results = svm.decision_function(embedding.embedding_)
    tn, fp, fn, tp = confusion_matrix(labels, [i > 0 for i in svm_results]).ravel()
    precision = tp / (tp + fp); recall = tp / (tp + fn); accuracy = (tp + tn) / len(labels)
    # print(f"accuracy: {accuracy:.2f} | precision: {precision:.2f} | recall: {recall:.2f}")
    #now, in [DESC15:4.2.1], they compare the "ranking induced by \vec{v_t} with the number of times the term occurs in the entity's documents" with Cohen's Kappa.
    num_occurances = [descriptions[ind].count_phrase(term) if occurs else 0 for ind, occurs in enumerate(labels)]
    #see notebooks/proof_of_concept/get_svm_decisionboundary.ipynb#Checking-projection-methods-&-distance-measures-from-point-to-projection for the ranking
    decision_plane = NDPlane(svm.coef_[0], svm.intercept_[0])
    dist = lambda x, plane: np.dot(plane.normal, x) + plane.intercept #TODO don't even need plane for this, just svm's stuff
    distances = [dist(point, decision_plane) for point in embedding.embedding_]
    argsort = sorted(enumerate(distances), key=lambda x:x[1])
    pos_rank = {item: rank for rank, item in enumerate([i[0] for i in argsort if i[1] > 0])}
    # compared = [(elem, pos_rank.get(ind)) for ind, elem in enumerate(num_occurances)]
    kappa = cohen_kappa_score(num_occurances, [pos_rank.get(ind, 0) for ind in range(len(num_occurances))])
    # print(f"Kappa-Score: {kappa}")
    if plot_svm:
        display_svm(embedding.embedding_, np.array(labels, dtype=int), svm, term=term, descriptions=descriptions, name=f"{term}: accuracy: {accuracy:.2f} | precision: {precision:.2f} | recall: {recall:.2f} | kappa: {kappa:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "kappa": kappa}, decision_plane



def display_svm(X, y, svm, term=None, name=None, descriptions=None):
    assert X.shape[1] == 3
    decision_plane = ThreeDPlane(svm.coef_[0], svm.intercept_[0])
    occurences = [descriptions[i].count_phrase(term) for i in range(len(X))]
    percentile = lambda percentage: np.percentile(np.array([i for i in occurences if i]), percentage)
    extras = [{"Name": descriptions[i].for_name, "Occurences": occurences[i], "extra": {"Description": shorten(descriptions[i].text, 200)}} for i in range(len(X))]
    with ThreeDFigure(name=name) as fig:
        fig.add_markers(X[np.where(np.logical_not(y))], color="blue", size=2, custom_data=[extras[i] for i in np.where(np.logical_not(y))[0]], linelen_right=50, name="negative samples")
        fig.add_markers(X[np.where(y)], color="red", size=[9 if occurences[i] > percentile(70) else 4 for i in np.where(y)[0]], custom_data=[extras[i] for i in np.where(y)[0]], linelen_right=50, name="positive samples")
        fig.add_surface(decision_plane, X, y, color="gray")  # decision hyperplane
        fig.add_line(X.mean(axis=0)-decision_plane.normal*2, X.mean(axis=0)+decision_plane.normal*2, width=2, name="Orthogonal")  # orthogonal of decision hyperplane through mean of points
        fig.add_markers([0, 0, 0], size=3, name="Coordinate Center")  # coordinate center
        # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        fig.show()