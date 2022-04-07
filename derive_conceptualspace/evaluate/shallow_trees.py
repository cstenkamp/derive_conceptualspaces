import subprocess
from collections import Counter
import warnings
import os
from os.path import abspath
import re
from contextlib import nullcontext

from misc_util.pretty_print import pretty_print as isnotebook
if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.metrics import get_scorer
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

from derive_conceptualspace.semantic_directions.cluster_names import get_name_dict
from derive_conceptualspace.settings import get_setting
from misc_util.pretty_print import pretty_print as print, isnotebook, display


#TODO write a test that checks if a decision-tree without test-set and with depth=None has a 1.0 accuracy:
# `classify_shallowtree(clusters, embedding, descriptions, one_vs_rest=False, dt_depth=None, test_percentage_crossval=0, classes="Geonames", do_plot=False, verbose=False)`

def classify_shallowtree_multi(clusters, embedding, descriptions, dataset_class, classes=None, verbose=False, **kwargs):
    results = {}
    for classes in (([classes] if isinstance(classes, str) else classes) or descriptions.additionals_names):
        for test_percentage_crossval in [0.33, 0.5, 4, 5]:
            for one_vs_rest in [True, False]:
                for dt_depth in [1,2,3,None]:
                    for balance_classes in [True, False]:
                        results[(classes, test_percentage_crossval, one_vs_rest, dt_depth, balance_classes)] = {}
                        for metric in ["accuracy", "f1"]:
                            print("=="*50)
                            score = classify_shallowtree(clusters, embedding, descriptions, dataset_class, one_vs_rest, dt_depth, test_percentage_crossval,
                                                         classes, verbose=verbose, return_features=False, balance_classes=balance_classes, metric=metric, **kwargs)
                            results[(classes, test_percentage_crossval, one_vs_rest, dt_depth, balance_classes)][metric] = score
    df = pd.DataFrame(results, columns=pd.MultiIndex.from_tuples([i for i in results.keys()], names=("classes","test%/Xval","1vsRest","Tree-Depth","balanced")))
    with pd.option_context('display.max_rows', 51, 'display.max_columns', 20, 'display.expand_frame_repr', False,
                           'display.max_colwidth', 20, 'display.float_format', '{:.4f}'.format):
        print(df)
    return df


def classify_shallowtree(clusters, embedding, descriptions, dataset_class, one_vs_rest, dt_depth, test_percentage_crossval,
                         classes=None, cluster_reprs=None, do_plot=False, verbose=False, return_features=True, metric="acc",
                         balance_classes=True, clus_rep_algo=None, shutup=False, repeat=1, pgbar=None, **kwargs):
    cm = (lambda *args, **kwargs: nullcontext()) if pgbar is None else tqdm
    clusters, planes = clusters.values()
    if classes is None:
        classes = descriptions.additionals_names[0]
    if classes in descriptions.additionals_names:
        catnames = None
        if hasattr(dataset_class, "CATNAMES") and classes in dataset_class.CATNAMES:
            catnames = dataset_class.CATNAMES.get(classes)
        hascat = [n for n,i in enumerate(descriptions._descriptions) if i._additionals[classes] is not None]
        getcat = lambda i: descriptions._descriptions[i]._additionals[classes]
    elif hasattr(dataset_class, "get_custom_class"):
        getcat, hascat, catnames = dataset_class.get_custom_class(classes, descriptions, verbose=(verbose and not shutup), **kwargs)
    else:
        raise Exception(f"The class {classes} does not exist!")
    if catnames:
        orig_getcat = getcat; getcat = lambda x: catnames.get(int(orig_getcat(x)), orig_getcat(x))

    if not shutup:
        print(f"Using classes from {classes} - {len(hascat)}/{len(descriptions)} entities have a class")
    cats = {i: getcat(i) for i in hascat}

    cluster_names = get_name_dict(clusters, cluster_reprs, clus_rep_algo)
    #first I want the distances to the origins of the respective dimensions (induced by the clusters), what induces the respective rankings (see DESC15 p.24u, proj2 of load_semanticspaces.load_projections)
    axis_dists = {i: {cluster_names[k]: v.dist(embedding[i]) for k, v in planes.items()} for i in hascat}
    if verbose and not shutup:
        best_per_dim = {k: descriptions._descriptions[v].title for k, v in pd.DataFrame(axis_dists).T.idxmax().to_dict().items()}
        print("Highest-ranking descriptions [with any class] per dimension:\n    "+"\n    ".join([f"*b*{k.ljust(max([len(i) for i in best_per_dim.keys()][:20]))}*b*: {v}" for k, v in best_per_dim.items()][:20]))
    #TODO also show places 2, 3, 4 - hier sehen wir wieder sehr ähnliche ("football stadium", "stadium", "fan" for "goalie")
    #TODO axis_dists is all I need for the movietuner already!! I can say "give me something like X, only with more Y"

    if not shutup:
        print(f"Labels ({len(set(cats.values()))} classes):", ", ".join(f"*b*{k}*b*: {v}" for k, v in Counter(cats.values()).items())) #TODO pay attention! consider class_weight etc!
    consider = pd.DataFrame({descriptions._descriptions[i].title: axis_dists[i] for i in hascat})
    ranked = pd.DataFrame([rankdata(i) for i in consider.values], index=consider.index, columns=consider.columns).astype(int).T
    ranked = ranked / ranked.shape[0] #looks better if we're doing relative rankings
    #TODO Teilweise sind Dinge ja in mehreren Klassen, da muss ich dann ja mehrere trees pro class machen!
    if not shutup:
        print(f'Eval-Settings: metric: *b*{metric}*b*, type: *b*{("one-vs-rest" if one_vs_rest else "all-at-once")}*b*, DT-Depth: *b*{dt_depth}*b*, train-test-split:*b*',
              f'{test_percentage_crossval}-fold cross-validation' if test_percentage_crossval > 1 else f'{test_percentage_crossval*100:.1f}% in test-set', "*b*")
    class_percents = sorted([i / len(cats) for i in Counter(cats.values()).values()], reverse=True)
    features_outvar = []
    all_targets = []
    all_classes = []
    if one_vs_rest:
        scores = {}
        plottree_strs = []
        raw_scores = {}
        with cm(total=len(set(cats.values()))*repeat, desc=pgbar if isinstance(pgbar, str) else None) as pgbar:
            for cat in set(cats.values()):
                targets = np.array(np.array(list(cats.values())) == cat, dtype=int)
                for n in range(repeat):
                    scores[cat], plottree_str, rawscores = classify(ranked.values, targets, list(ranked.columns), ["other", cat], dt_depth, test_percentage_crossval, metric=metric,
                                                            do_plot=do_plot, features_outvar=features_outvar, balance_classes=balance_classes, do_render=False, shuffle=(repeat>1))
                    raw_scores.setdefault(cat, []).append(rawscores)
                    if pgbar is not None: pgbar.update(1)
                plottree_strs.append(plottree_str)
                all_targets.append(targets)
                all_classes.append(["other", cat])
        if do_plot and not shutup: #holy shit that merging took a while - see https://stackoverflow.com/q/47258673/5122790 for how
            dot_cnts = [subprocess.run(["dot"], stdout=subprocess.PIPE, input=str(i), encoding="UTF-8").stdout for n, i in enumerate(plottree_strs)]
            if not isnotebook():
                a = subprocess.run(["dot"], stdout=subprocess.PIPE, input="\n".join(dot_cnts), encoding="UTF-8").stdout
                b = subprocess.run(["gvpack", "-array_t2", "-m20"], stdout=subprocess.PIPE, input=a, encoding="UTF-8").stdout
                b = re.sub(r"<br\/>value = \[.*?\]", "", b)
                b = re.sub(r" &le; 0.(\d\d)(\d*)<br", r" &le; \1.\2% <br", b)
                subprocess.run(["neato", "-n2", "-s", "-Tpdf", "-o", "merge.pdf"], stdout=subprocess.PIPE, input=b, encoding="UTF-8")
                print(f"Saved under {abspath('merge.pdf')}")
                os.system("xdg-open merge.pdf")
            else:
                JUPYTER_N_COLS = 2
                plots = [subprocess.run(["gvpack", "-m50"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, input="\n".join(dot_cnts[i:i+JUPYTER_N_COLS]), encoding="UTF-8").stdout for i in range(0, len(dot_cnts), JUPYTER_N_COLS)]
                for plot in plots:
                    display(graphviz.Source(plot))
                    print("\n\n")
        if repeat > 1 and isinstance(metric, (list, tuple)): #this is dirty as fuck but whelp, time is getting knapp
            return raw_scores, Counter(cats.values())
        score = sum([v * Counter(cats.values())[k] for k, v in scores.items()]) / len(cats)
        if not shutup:
            print("Per-Class-Scores:", ", ".join(f"{k}: {v:.2f}" for k, v in scores.items()))
            print(f"Unweighted Mean {metric}: {sum(scores.values()) / len(scores.values()):.2%}")
            print(f"Weighted Mean {metric}: {score:.2%}")
    else: #all at once
        raw_scores = []
        if dt_depth is not None and len(set(cats.values())) > 2**dt_depth:
            warnings.warn(f"There are more classes ({len(set(cats.values()))}) than your decision-tree can possibly classify ({2**dt_depth})")
        targets = np.array(list(cats.values()))
        all_targets.append(targets)
        for n in range(repeat):
            score,_,rawscores = classify(ranked.values, targets, list(ranked.columns), list(catnames.values()), dt_depth, test_percentage_crossval, metric=metric,
                               do_plot=do_plot, features_outvar=features_outvar, balance_classes=balance_classes, do_render=False)
            raw_scores.append(rawscores)
            all_classes.append(list(catnames.values()))
        if not shutup and repeat <= 1:
            print(f"{metric}: {score:.2f}")
            if dt_depth == 1:
                print(f"Baseline {metric}: {class_percents[0]:.2f}") #all into one class. Praxis is often worse than this because we have class_weight=balanced.
                if metric.lower() in ["acc", "accuracy"]:
                    print(f"Maximally achievable {metric}: {sum(class_percents[:2]):.2f}") #two leaves, one is the (perfectly classified) class 1, the other get's the label for the second-most-common class
    if repeat > 1:
        return raw_scores, Counter(cats.values())
    if return_features:
        return [features_outvar, ranked, all_targets, list(scores.values()) if one_vs_rest else [score], all_classes, list(ranked.columns)]
    return score


def classify(input, target, axnames, catnames, dt_depth, test_percentage_crossval, metric,
             do_plot=False, features_outvar=None, balance_classes=True, do_render=False, shuffle=False):
    # input[:, 99] = (target == "Shops&Services"); axnames[99] = "is_shop"
    # input[:, 98] = (target == "Food"); axnames[98] = "is_food"
    metric = "accuracy" if metric == "acc" else metric #https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    kwargs = dict(class_weight="balanced") if balance_classes else {}
    clf = DecisionTreeClassifier(random_state=get_setting("RANDOM_SEED"), max_depth=dt_depth, **kwargs)
    if test_percentage_crossval > 1:
        if metric == "f1" and len(catnames) > 2:
            metric = "f1_micro"
        cv = test_percentage_crossval if not shuffle else StratifiedKFold(n_splits=test_percentage_crossval, shuffle=True)
        #see "cv" parameter of cross_val_score at https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
        if isinstance(metric, (list, tuple)):
            if len(catnames) > 2:
                metric = [i if i != "f1" else "f1_macro" for i in metric] #f1_macro would be = accuracy
            return None, None, cross_validate(clf, input, target, cv=cv, scoring=metric)
        else:
            scores = cross_val_score(clf, input, target, cv=cv, scoring=metric)
        score = scores.mean()
        clf.fit(input, target)
        if metric == "accuracy":
            assert get_score(clf, input, target, metric) == np.array([res==target[i] for i, res in enumerate(clf.predict(input))]).mean()
        else:
            get_score(clf, input, target, metric, is_multiclass=len(catnames) > 2) #have to to be able to plot_tree
        # print(f"Doing {test_percentage_crossval}-fold cross-validation. Best Score: {scores.max():.2f}, Mean: {score}:.2f")
    elif test_percentage_crossval == 0:
        warnings.warn("Using the full data as training set without a test-set!")
        clf.fit(input, target)
        score = scores = get_score(clf, input, target, metric, is_multiclass=len(catnames) > 2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=test_percentage_crossval) #TODO: stratify? https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        clf.fit(X_train, y_train)
        score = scores = get_score(clf, X_test, y_test, metric, is_multiclass=len(catnames) > 2)
    if features_outvar is not None:
        features_outvar.append(clf)
    if do_plot:
        if catnames: assert len(clf.classes_) == len(catnames)
        return score, plot_tree(clf, axnames, (catnames or [str(i) for i in clf.classes_]), do_render=do_render), scores
    return score, None, scores

def get_score(clf, X_test, y_test, metric, is_multiclass=False):
    if metric == "acc":
        return clf.score(X_test, y_test)
    if is_multiclass and metric == "f1":
        metric = "f1_micro" #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    return get_scorer(metric)(clf, X_test, y_test)




def plot_tree(clf, axnames=None, catnames=None, do_render=False):
    kwargs = {}
    if axnames is not None: kwargs.update(feature_names=axnames)
    if catnames is not None: kwargs.update(class_names=[i.replace("&", " and ") for i in catnames])
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, impurity=False, **kwargs)
    graph = graphviz.Source(dot_data)
    if do_render:
        graph.render("arg", view=True)
    return graph
