from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from HanTa import HanoverTagger as ht

def tokenize_sentences_countvectorizer(descriptions):
    #TODO CountVectorizer can be customized a lot, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer()
    counted = vectorizer.fit_transform(descriptions)
    vocab = vectorizer.get_feature_names()
    return vocab, counted.toarray()

def tokenize_sentences_nltk(descriptions):
    #https://textmining.wp.hs-hannover.de/Preprocessing.html#Satzerkennung-und-Tokenization
    sent_list = [nltk.sent_tokenize(x, language="german") for x in descriptions]
    #so as we're really only concerning bags-of-words here, we run a lemmatizer
    # (see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung)
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    res = []
    words = set()
    for n, sample in tqdm(enumerate(sent_list)):
        #TODO: recognize language ^^
        all_tags = []
        for sent in sample:
            tokenized_sent = nltk.tokenize.word_tokenize(sent, language='german')
            tags = tagger.tag_sent(tokenized_sent)
            all_tags.extend([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
        res.append(all_tags) # we could res.append(Counter(all_tags))
        words.update(all_tags)
    words = list(words)
    alls = []
    for wordlist in res:
        cnt = Counter(wordlist)
        alls.append(np.array([cnt[i] for i in words]))
    return words, np.array(alls)