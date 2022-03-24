import random
from collections import Counter

import gensim
import numpy as np
from tqdm import tqdm
from keybert import KeyBERT

flatten = lambda l: [item for sublist in l for item in sublist]


class KeybertRepr():
    def __init__(self, lang):
        if lang == "de":
            self.model = KeyBERT("dbmdz/bert-base-german-uncased")
        else:
            raise NotImplementedError()

    def get_repr(self, lst, n_shuffles=5, max_ngram=None):
        trials = []
        for n_comb in range(n_shuffles):
            lst = random.sample(lst, len(lst))
            cands = lst
            if max_ngram is not None:
                cands = [i for i in lst if i.count(" ") < max_ngram]
                if not cands: cands = lst
            trials.append(self.model.extract_keywords(". ".join(lst), candidates=cands, top_n=1)[0])
        if len(set(i[0] for i in trials)) == 1:
            return trials[0][0]
        elif (cnt := sorted(Counter([i[0] for i in trials]).values(), reverse=True))[0] > cnt[1]: #if one is detected more often than the others
            return max(Counter([i[0] for i in trials]), key=lambda x: x[1])
        #now: return the one that is most often extracted and then the highest score from that
        trials = [j for j in trials if Counter([i[0] for i in trials])[j[0]] == max(Counter([i[0] for i in trials]).values())]
        return max(trials, key=lambda x:x[1])[0]

class GensimRepr():
    def __init__(self, model_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.init_replacers()

    def init_replacers(self):
        changeds = {}
        for key in tqdm(self.model.key_to_index.keys(), desc="Making Umlaut-Replacers"):
            k2 = key
            for uml in "uoa":
                k2 = k2.replace(uml + "e", uml)
            if k2 != key:
                changeds[k2] = key
        self.replacers = {k.lower(): v for k, v in changeds.items()}

    def get_repr(self, lst, weights=None):
        embeddings = [[self.model[j] if j in self.model else
                        self.model[j.capitalize()] if j.capitalize() in self.model else
                          self.model[self.replacers[j]] if j in self.replacers else None
                        for j in i.split(" ")] for i in lst]
        embeddings = [np.array([j for j in i if j is not None]).mean(axis=0) if i and not all(j is None for j in i) else np.nan for i in embeddings]
        if weights is not None:
            embeddings = [emb*weight for emb, weight in zip(embeddings, weights)]
        embeddings = [i for i in embeddings if not np.isnan(i).all()]
        # onegrams = [trained_model.most_similar(i)[0][0] for i in embeddings]
        if len(embeddings) == 0:
            return lst[0]
        return self.model.similar_by_vector(np.array(embeddings).mean(axis=0))[0][0].replace("_"," ").lower()


def get_cluster_reprs(clusters, featureaxes, filtered_dcm, metric, model_path, lang):
    res = {}
    gnsim = GensimRepr(model_path)
    kbert = KeybertRepr(lang)
    # clusters = list(clusters.items())[:20]
    for k, v in tqdm(clusters.items(), desc="Finding Cluster Representations"):
        lst = [k]+v
        res[k] = dict(keybert    = kbert.get_repr(lst),
                      keybert_1g = kbert.get_repr(lst, max_ngram=1),
                      gensim     = gnsim.get_repr(lst),
                      gensim_w1  = gnsim.get_repr(lst, weights=[np.mean([filtered_dcm.term_freq(j, relative=True) for j in i.split(" ")]) for i in lst]),
                      gensim_w2  = gnsim.get_repr(lst, weights=[featureaxes["metrics"][i]["f_one"] for i in lst]),
                      gensim_w3  = gnsim.get_repr(lst, weights=[featureaxes["metrics"][i][metric] for i in lst]),
                      )
    return res

