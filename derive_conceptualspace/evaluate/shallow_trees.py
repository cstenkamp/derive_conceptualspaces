from collections import Counter
import warnings

import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import rankdata

from derive_conceptualspace.settings import get_setting
from misc_util.pretty_print import pretty_print as print

CATNAMES = {
    "Geonames": {1: "stream,lake", 2: "parks,area", 3: "road,railroad", 4: "spot,building,farm", 5: "mountain,hill,rock", 6: "undersea", 7: "forest,heath"}, #can recover from http://www.geonames.org/export/codes.html
    "Foursquare": {1: "Arts&Entertainment", 2: "College&University", 3: "Food", 4: "Professional&Other", 5: "NightlifeSpots", 6: "GreatOutdoors", 7: "Shops&Services", 8:"Travel&Transport", 9:"Residences"}, #https://web.archive.org/web/20140625051659/http://aboutfoursquare.com/foursquare-categories/
}
#TODO move somewhere appropriate (-> dataset-class)


#TODO write a test that checks if a decision-tree without test-set and with depth=None has a 1.0 accuracy:
# `classify_shallowtree(clusters, embedding, descriptions, one_vs_rest=False, dt_depth=None, test_percentage_crossval=0, classes="Geonames", do_plot=False, verbose=False)`

def classify_shallowtree_multi(clusters, embedding, descriptions, verbose=False):
    results = {}
    for classes in descriptions.additionals_names:
        for test_percentage_crossval in [0.33, 0.5, 4, 5]:
            for one_vs_rest in [True, False]:
                for dt_depth in [1,2,3]:
                    for balance_classes in [True, False]:
                        print("=="*50)
                        score = classify_shallowtree(clusters, embedding, descriptions, one_vs_rest, dt_depth, test_percentage_crossval, classes, do_plot=False, verbose=False, return_features=False, balance_classes=balance_classes)
                        results[(classes, test_percentage_crossval, one_vs_rest, dt_depth, balance_classes)] = score
    df = pd.DataFrame(results, index=["Accuracy"], columns=pd.MultiIndex.from_tuples([i for i in results.keys()], names=("classes","test%/Xval","1vsRest","Tree-Depth","balanced"))).T
    with pd.option_context('display.max_rows', 51, 'display.max_columns', 20, 'display.expand_frame_repr', False,
                           'display.max_colwidth', 20, 'display.float_format', '{:.4f}'.format):
        print(df)
    return df

def classify_shallowtree(clusters, embedding, descriptions, one_vs_rest, dt_depth, test_percentage_crossval,
                         classes=None, do_plot=False, verbose=False, return_features=True, balance_classes=True):
    clusters, planes = clusters.values()
    if classes is None:
        classes = descriptions.additionals_names[0]
    else:
        assert classes in descriptions.additionals_names
    catnames = CATNAMES.get(classes) #TODO if it's in, otherwise yadda

    #first I want the distances to the origins of the respective dimensions (induced by the clusters), what induces the respective rankings (see DESC15 p.24u, proj2 of load_semanticspaces.load_projections)
    axis_dists = [{k: v.dist(embedding[i]) for k, v in planes.items()} for i in range(len(embedding))]
    best_per_dim = {k: descriptions._descriptions[v].title for k, v in pd.DataFrame(axis_dists).idxmax().to_dict().items()}
    if verbose:
        print("Highest-ranking descriptions per dimension:\n    "+"\n    ".join([f"{k.ljust(max([len(i) for i in best_per_dim.keys()][:20]))}: {v}" for k, v in best_per_dim.items()][:20]))
    #TODO also show places 2, 3, 4 - hier sehen wir wieder sehr ähnliche ("football stadium", "stadium", "fan" for "goalie")
    #TODO axis_dists is all I need for the movietuner already!! I can say "give me something like X, only with more Y"
    if catnames:
        cats = {i.title: catnames[int(i._additionals[classes])] for i in descriptions._descriptions if i._additionals[classes] is not None}
    else:
        cats = {i.title: i._additionals[classes] for i in descriptions._descriptions if i._additionals[classes] is not None}
    print(f"Using classes from {classes} - {len(cats)}/{len(descriptions)} entities have a class")
    with_cat = [n for n, i in enumerate(descriptions._descriptions) if i._additionals[classes] is not None]
    print("Labels:", ", ".join(f"{k}: {v}" for k, v in Counter(cats.values()).items())) #TODO pay attention! consider class_weight etc!
    consider = pd.DataFrame({descriptions._descriptions[i].title: axis_dists[i] for i in with_cat})
    ranked = pd.DataFrame([rankdata(i) for i in consider.values], index=consider.index, columns=consider.columns, dtype=int).T
    axnames = [f",".join([i]+clusters[i][:2]) for i in ranked.columns] #TODO would be better if done from the mentioned ways
    #TODO Teilweise sind Dinge ja in mehreren Klassen, da muss ich dann ja mehrere trees pro class machen!
    print(f'Eval-Settings: type: *b*{("one-vs-rest" if one_vs_rest else "all-at-once")}*b*, DT-Depth: *b*{dt_depth}*b*, train-test-split:*b*',
          f'{test_percentage_crossval}-fold cross-validation' if test_percentage_crossval > 1 else f'{test_percentage_crossval*100:.1f}% in test-set', "*b*")
    class_percents = sorted([i / len(cats) for i in Counter(cats.values()).values()], reverse=True)
    features_outvar = []
    all_targets = []
    all_classes = []
    if one_vs_rest:
        scores = {}
        for cat in set(cats.values()):
            targets = np.array(np.array(list(cats.values())) == cat, dtype=int)
            scores[cat] = classify(ranked.values, targets, axnames, ["other", cat], dt_depth, test_percentage_crossval, do_plot=do_plot, features_outvar=features_outvar, balance_classes=balance_classes)
            all_targets.append(targets)
            all_classes.append(["other", cat])
        print("Per-Class-Scores:", ", ".join(f"{k}: {v:.2f}" for k, v in scores.items()))
        print(f"Unweighted Mean Accuracy: {sum(scores.values()) / len(scores.values()):.2f}")
        score = sum([v*Counter(cats.values())[k] for k, v in scores.items()])/len(cats)
        print(f"Weighted Mean Accuracy: {score:.2f}")
    else: #all at once
        if dt_depth is not None and len(set(cats.values())) > 2**dt_depth:
            warnings.warn(f"There are more classes ({len(set(cats.values()))}) than your decision-tree can possibly classify ({2**dt_depth})")
        targets = np.array(list(cats.values()))
        all_targets.append(targets)
        score = classify(ranked.values, targets, axnames, list(catnames.values()), dt_depth, test_percentage_crossval, do_plot=do_plot, features_outvar=features_outvar, balance_classes=balance_classes)
        all_classes.append(list(catnames.values()))
        print(f"Accuracy: {score:.2f}")
        if dt_depth == 1:
            print(f"Baseline Accuracy: {class_percents[0]:.2f}") #all into one class. Praxis is often worse than this because we have class_weight=balanced.
            print(f"Maximally achievable Accuracy: {sum(class_percents[:2]):.2f}") #two leaves, one is the (perfectly classified) class 1, the other get's the label for the second-most-common class
    if return_features:
        return [features_outvar, ranked, all_targets, list(scores.values()) if one_vs_rest else [score], all_classes]
    return score


def classify(input, target, axnames, catnames, dt_depth, test_percentage_crossval, do_plot=False, features_outvar=None, balance_classes=True):
    # input[:, 99] = (target == "Shops&Services"); axnames[99] = "is_shop"
    # input[:, 98] = (target == "Food"); axnames[98] = "is_food"
    kwargs = dict(class_weight="balanced") if balance_classes else {}
    clf = DecisionTreeClassifier(random_state=get_setting("RANDOM_SEED"), max_depth=dt_depth, **kwargs)
    if test_percentage_crossval > 1:
        scores = cross_val_score(clf, input, target, cv=test_percentage_crossval)
        score = scores.mean()
        clf.fit(input, target)
        assert clf.score(input, target) == np.array([res==target[i] for i, res in enumerate(clf.predict(input))]).mean()#have to to be able to plot_tree
        # print(f"Doing {test_percentage_crossval}-fold cross-validation. Best Score: {scores.max():.2f}, Mean: {score}:.2f")
    elif test_percentage_crossval == 0:
        warnings.warn("Using the full data as training set without a test-set!")
        clf.fit(input, target)
        score = clf.score(input, target)
    else:
        X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=test_percentage_crossval) #TODO: stratify? https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    if do_plot:
        plot_tree(clf, axnames, clf.classes_)
    if features_outvar is not None:
        features_outvar.append(clf)
    return score


def plot_tree(clf, axnames=None, catnames=None):
    kwargs = {}
    if axnames is not None: kwargs.update(feature_names=axnames)
    if catnames is not None: kwargs.update(class_names=[i.replace("&", " and ") for i in catnames])
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True, impurity=False, **kwargs)
    graph = graphviz.Source(dot_data)
    graph.render("arg", view=True)
