from functools import lru_cache
from math import log
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import unidecode
from collections import Counter
from functools import partial, lru_cache
from typing import List

import numpy as np
from tqdm import tqdm
import nltk
from HanTa import HanoverTagger as ht
from nltk.corpus import stopwords as nlstopwords

from derive_conceptualspace.util.desc_object import Description
from derive_conceptualspace.util.dtm_object import DocTermMatrix
from derive_conceptualspace.util.nltk_util import NLTK_LAN_TRANSLATOR, wntag
from derive_conceptualspace.util.tokenizers import tokenize_text

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
########################################################################################################################

def load_desc15_stopwords():
    with open(join(dirname(__file__), "stopwords_desc15.txt"), "r") as rfile:
        lines = rfile.readlines()
    lines = [i.strip() for i in lines if i.strip() and not i.strip().startswith("|")]
    return set((i if not "|" in i else i[:i.find("|")]).strip() for i in lines)


@lru_cache(maxsize=None)
def get_stopwords(language, include_desc15_stopwords=True):
    if language in NLTK_LAN_TRANSLATOR:
        language = NLTK_LAN_TRANSLATOR[language]
    assert language in NLTK_LAN_TRANSLATOR.values()
    stopwords = set(nlstopwords.words(language))
    if include_desc15_stopwords and language == "english":
        stopwords |= load_desc15_stopwords()
    return tuple(stopwords)


def run_preprocessing_funcs(descriptions:List[Description], components: dict, word_tokenizer=None):
    #components: sent_tokenize=True, lemmatize=True, remove_stopwords=True, convert_lower=True, remove_diacritcs=True, remove_punctuation=True
    # TODO use TreeTagger? https://textmining.wp.hs-hannover.de/Preprocessing.html#Alternative:-Treetagger
    # https://textmining.wp.hs-hannover.de/Preprocessing.html#Satzerkennung-und-Tokenization
    assert components.get("convert_lower"), "Stopwords are lower-case so not converting is not allowed (bad for german...!)"
    if components.get("sent_tokenize"):
        descriptions = sent_tokenize_all(descriptions)
    if components.get("convert_lower"):
        descriptions = convert_lower_all(descriptions)
    #tokenization will happen anyway!
    if not components.get("lemmatize"):
        descriptions = word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=components.get("remove_stopwords", False))
    else:
        descriptions = word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=False)
        descriptions = lemmatize_all(descriptions)
        for desc in descriptions:
            desc.process([[lemma for lemma in sent if lemma not in get_stopwords(desc.lang)] for sent in desc.processed_text], "remove_stopwords")
    if components.get("remove_diacritics"):
        descriptions = remove_diacritics_all(descriptions)
    if components.get("remove_punctuation"):
        descriptions = remove_punctuation_all(descriptions)
    return descriptions

def sent_tokenize_all(descriptions):
    for desc in descriptions:
        desc.process(nltk.sent_tokenize(desc.processed_text, language=NLTK_LAN_TRANSLATOR[desc.lang]), "sent_tokenize")
    return descriptions

def convert_lower_all(descriptions):
    for desc in descriptions:
        if isinstance(desc.processed_text, list):
            desc.process([i.lower() for i in desc.processed_text], "convert_lower")
        else:
            desc.process(desc.processed_text.lower(), "convert_lower")
    return descriptions

def word_tokenize_all(descriptions, word_tokenizer=None, remove_stopwords=False):
    print("Word-Tokenizing Descriptions...")
    tokenizer_fn_name = word_tokenizer.__name__ if word_tokenizer is not None else "tokenize_text"
    word_tokenizer = word_tokenizer or (lambda *args, **kwargs: tokenize_text(*args, **kwargs)[1])
    for desc in tqdm(descriptions):
        if isinstance(desc.processed_text, list):
            if remove_stopwords:
                desc.process([word_tokenizer(i, get_stopwords(desc.lang)) for i in desc.processed_text], tokenizer_fn_name+"_removestopwords")
            else:
                desc.process([word_tokenizer(i) for i in desc.processed_text], tokenizer_fn_name)
        else:
            if remove_stopwords:
                desc.process(word_tokenizer(desc.processed_text, get_stopwords(desc.lang)), tokenizer_fn_name+"_removestopwords")
            else:
                desc.process(word_tokenizer(desc.processed_text), tokenizer_fn_name)
    return descriptions

def lemmatize_all(descriptions, use_better_german_tagger=True):
    # see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung
    print("Start lemmatizing")
    german_tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    lemmatizer = nltk.WordNetLemmatizer()
    for desc in tqdm(descriptions):
        assert "sent_tokenize" in [i[1] for i in desc.processing_steps]
        assert not any("removestopwords" in i[1] for i in desc.processing_steps) #taggers work best on sentences
        lemmatized = []
        for sent in desc.processed_text:
            if desc.lang == "de" and use_better_german_tagger:
                if isinstance(sent, list):
                    sent = " ".join(sent) #TODO really?!
                tags = german_tagger.tag_sent(sent)
                lemmatized.append([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
                raise NotImplementedError()
            else:
                tags = nltk.pos_tag(sent)
                lemmatized.append([lemmatizer.lemmatize(word, wntag(pos)) if wntag(pos) else word for word, pos in tags])
        desc.process(lemmatized, "lemmatize")
    return descriptions


def remove_diacritics_all(descriptions):
    for desc in descriptions:
        if isinstance(desc.processed_text, list):
            if isinstance(desc.processed_text[0], list):
                desc.process([[unidecode.unidecode(word) for word in sent] for sent in desc.processed_text], "remove_diacritics")
            else:
                desc.process([unidecode.unidecode(sent) for sent in desc.processed_text], "remove_diacritics")
        else:
            desc.process(unidecode.unidecode(desc.processed_text), "remove_diacritics")
    return descriptions


def remove_punctuation(sentences):
    # desc.process([[word for word in sent if word.isalnum()] for sent in desc.processed_text], "remove_diacritics")
    condition = lambda letter: letter.isalnum() or letter in "-`"
    sents = []
    for sent in sentences:
        words = []
        for word in sent:
            if word.isalnum():
                words.append(word)
            else:
                if len(word) > 1:
                    if all(condition(letter) for letter in word):
                        words.append(word)
                    for letter in word:
                        if not condition(letter):
                            parts = [i for i in word.split(letter) if i]
                            if all(i.isalnum() for i in parts):
                                words.extend(parts)
                            # else:
                            #     print(f"Error at {word}") #TODO irgendwann muss ich mich darum kümmern
        sents.append(words)
    return sents


def remove_punctuation_all(descriptions):
    for desc in descriptions:
        if isinstance(desc.processed_text, list):
            if isinstance(desc.processed_text[0], list):
                sents = remove_punctuation(desc.processed_text)
                desc.process(sents, "remove_punctuation")
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    return descriptions


def make_bow(descriptions):
    all_words = sorted(set(flatten([flatten(desc.processed_text) for desc in descriptions])))
    for desc in descriptions:
        desc.bow = Counter(flatten(desc.processed_text))
    return all_words, descriptions



# def tokenize_sentences_nltk(descriptions):
#     #so as we're really only concerning bags-of-words here, we run a lemmatizer
#     # (see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung)
#     tagger = ht.HanoverTagger('morphmodel_ger.pgz')
#     res = []
#     words = set()
#     for n, sample in enumerate(tqdm(descriptions)):
#         all_tags = []
#         assert "sent_tokenize" in [i[1] for i in sample.processing_steps]
#         for sent in sample.processed_text:
#             tags = tagger.tag_sent(sent)
#             all_tags.extend([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
#         res.append(all_tags) # we could res.append(Counter(all_tags))
#         words.update(all_tags)
#     words = list(words)
#     alls = []
#     for wordlist in res:
#         cnt = Counter(wordlist)
#         alls.append(np.array([cnt[i] for i in words]))
#     return words, np.array(alls)





########################################################################################################################
########################################################################################################################
########################################################################################################################

#TODO use tf-idf as alternative keyword-detection! (erst mit gensim.dictionary alle Wörter holen, dann tf-idf drauffwerfen)
def tf_idf(doc_term_matrix, verbose=False, mds_obj=None, descriptions=None):
    #TODO make this and ppmi equal!
    """see https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016"""
    n_docs = len(doc_term_matrix.dtm)
    quantifications = [[[term, count * log(n_docs/doc_term_matrix.doc_freqs[term])] for term, count in doc] for doc in doc_term_matrix.dtm]
    if verbose:
        print("Running TF-IDF on the corpus...")
        print_quantification(doc_term_matrix, quantifications, mds_obj, descriptions)
    quantifications = DocTermMatrix(dict(doc_term_matrix=quantifications, all_terms=doc_term_matrix.all_terms))
    return quantifications


def print_quantification(dtm, quantifications, mds_obj=None, descriptions=None):
    if mds_obj:
        getname = lambda id: mds_obj.names[id]
    elif descriptions:
        getname = lambda id: descriptions[id].for_name
    else:
        assert False, "if verbose you need to provide either mds_obj or descriptions!"
    distinctive_terms = [max(doc, key=lambda x:x[1]) if doc else [-1, 0] for doc in quantifications]
    most_distinct = sorted([[ind, elem] for ind, elem in enumerate(distinctive_terms)], key=lambda x:x[1][1], reverse=True)[:10]
    print("Most distinct terms:\n  "+"\n  ".join([f"`{dtm.all_terms[termid].ljust(12)}` ({str(round(value)).ljust(3)}) in `{getname(docid)}`" for docid, (termid, value) in most_distinct]))
    frequent_words = [[dtm.all_terms[i[0]], i[1]] for i in sorted([[k, v] for k, v in dtm.doc_freqs.items()], key=lambda x: x[1], reverse=True)[:20]]
    print("Terms that are in many documents:", ", ".join([f"{i[0]} ({round(i[1]/len(dtm.dtm)*100)}%)" for i in frequent_words]))
    values_per_phrase = {}
    for phrase, value in flatten(quantifications):
        values_per_phrase.setdefault(phrase, []).append(value)
    average_phrase_val = {k: sum(v)/len(v) for k, v in values_per_phrase.items()}
    worst_phrases = [dtm.all_terms[i[0]] for i in sorted([[k,v] for k,v in average_phrase_val.items()], key=lambda x:x[1])[:20]]
    print("Keyphrases with the lowest average score:", ", ".join(worst_phrases))
    #TODO maybe have an option to sort these out? But on the other hand, if half the courses contain the words `basic` or `introduction`, that's worth something
    #TODO alternatively also remove those terms with a high document frequency? important for LDA (aber dann brauch ich ne manuelle liste an to-keep (ich pack ja "seminar" explicitly rein))
