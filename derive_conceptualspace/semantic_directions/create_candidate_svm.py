from derive_conceptualspace.settings import CANDIDATETERM_MIN_OCCURSIN_DOCS
from derive_conceptualspace.util.jsonloadstore import json_load
from tqdm import tqdm
import sklearn.svm
import numpy as np

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from derive_conceptualspace.util.base_changer import Plane, make_base_changer
from derive_conceptualspace.util.threedfigure import ThreeDFigure, make_meshgrid



def create_candidate_svms(dcm, mds, descriptions, verbose):
    metrics = {}
    for term, exist_indices in tqdm(dcm.term_existinds(use_index=False).items()):
        assert len(exist_indices) >= CANDIDATETERM_MIN_OCCURSIN_DOCS
        metrics[term] = create_candidate_svm(mds, term, exist_indices, descriptions)
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


def create_candidate_svm(mds, term, exist_indices, descriptions):
    #TODO [DESC15]: "we adapted the costs of the training instances to deal with class imbalance (using the ratio between entities with/without the term as cost)"
    # TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
    labels = [False] * mds.embedding_.shape[0]
    for i in exist_indices:
        labels[i] = True
    svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
    svm.fit(mds.embedding_, np.array(labels, dtype=int))
    svm_results = svm.decision_function(mds.embedding_)
    tn, fp, fn, tp = confusion_matrix(labels, [i > 0 for i in svm_results]).ravel()
    precision = tp / (tp + fp); recall = tp / (tp + fn); accuracy = (tp + tn) / len(labels)
    # print(f"accuracy: {accuracy:.2f} | precision: {precision:.2f} | recall: {recall:.2f}")
    if term in ["component", "theory"] and mds.embedding_.shape[1] == 3:
        display_svm(mds.embedding_, np.array(labels, dtype=int), svm, name=term)

    #now, in [DESC15:4.2.1], they compare the "ranking induced by \vec{v_t} with the number of times the term occurs in the entity's documents" with Cohen's Kappa.
    num_occurances = [descriptions[ind].count_phrase(term) if occurs else 0 for ind, occurs in enumerate(labels)]
    #see notebooks/proof_of_concept/get_svm_decisionboundary.ipynb#Checking-projection-methods-&-distance-measures-from-point-to-projection for the ranking
    decision_plane = Plane(*svm.coef_[0], svm.intercept_[0])
    dist = lambda x, plane: np.dot(plane.normal, x) + plane.d #TODO don't even need plane for this, just svm's stuff
    distances = [dist(point, decision_plane) for point in mds.embedding_]
    argsort = sorted(enumerate(distances), key=lambda x:x[1])
    pos_rank = {item: rank for rank, item in enumerate([i[0] for i in argsort if i[1] > 0])}
    # compared = [(elem, pos_rank.get(ind)) for ind, elem in enumerate(num_occurances)]
    kappa = cohen_kappa_score(num_occurances, [pos_rank.get(ind, 0) for ind in range(len(num_occurances))])
    # print(f"Kappa-Score: {kappa}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "kappa": kappa}



def display_svm(X, y, svm, name=None):
    assert X.shape[1] <= 3
    decision_plane = Plane(*svm.coef_[0], svm.intercept_[0])
    with ThreeDFigure(name=name) as fig:
        fig.add_markers(X[np.where(y)], color="red", size=2, name="positive samples")  # samples
        fig.add_markers(X[np.where(~y)], color="blue", size=2, name="negative samples")  # samples
        fig.add_surface(decision_plane, X, y, color="gray")  # decision hyperplane
        fig.add_line(X.mean(axis=0)-decision_plane.normal*2, X.mean(axis=0)+decision_plane.normal*2, width=2)  # orthogonal of decision hyperplane through mean of points
        fig.add_markers([0, 0, 0], size=3)  # coordinate center
        # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        fig.show()