import gc
import logging
from functools import partial, lru_cache
from math import log
from os.path import join, dirname, basename
import re

import nltk
import numpy as np
import unidecode
from HanTa import HanoverTagger as ht
from nltk.corpus import stopwords as nlstopwords
from tqdm import tqdm
import scipy.sparse.csr

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.util.dtm_object import csr_to_list
from derive_conceptualspace.util.nltk_util import NLTK_LAN_TRANSLATOR, wntag
from derive_conceptualspace.util.np_tools import np_divide, np_log
from derive_conceptualspace.util.tokenizers import tokenize_text

logger = logging.getLogger(basename(__file__))

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
def get_stopwords(language, include_desc15_stopwords=True, include_custom=True):
    if language in NLTK_LAN_TRANSLATOR:
        language = NLTK_LAN_TRANSLATOR[language]
    assert language in NLTK_LAN_TRANSLATOR.values(), f"Cannot deal with language {language}"
    stopwords = set(nlstopwords.words(language))
    if include_desc15_stopwords and language == "english":
        stopwords |= load_desc15_stopwords()
    if include_custom and language == "english":
        stopwords |= set(get_setting("CUSTOM_STOPWORDS"))
    return tuple(stopwords)


def run_preprocessing_funcs(descriptions:DescriptionList, components, word_tokenizer=None):
    #components: sent_tokenize=True, lemmatize=True, remove_stopwords=True, convert_lower=True, remove_diacritcs=True, remove_punctuation=True
    # TODO use TreeTagger? https://textmining.wp.hs-hannover.de/Preprocessing.html#Alternative:-Treetagger
    # https://textmining.wp.hs-hannover.de/Preprocessing.html#Satzerkennung-und-Tokenization
    assert components.convert_lower, "Stopwords are lower-case so not converting is not allowed (bad for german...!)"
    if components.remove_htmltags:
        descriptions.process_all(lambda data: re.compile(r'<.*?>').sub('', data), "remove_htmltags")
    if components.sent_tokenize:
        descriptions.process_all(nltk.sent_tokenize, "sent_tokenize", indiv_kwargs=dict(language=lambda desc: NLTK_LAN_TRANSLATOR[desc.lang]))
    if components.convert_lower:
        convert_lower_all(descriptions)
    #tokenization will happen anyway!
    if not components.lemmatize:
        word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=components.remove_stopwords)
    else:
        word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=False)
        lemmatize_all(descriptions)
        descriptions.process_all(lambda txt, stopwords: [[lemma for lemma in sent if lemma not in stopwords] for sent in txt], "remove_stopwords", indiv_kwargs=dict(stopwords=lambda desc: get_stopwords(desc.lang)))
    if components.remove_diacritics:
        remove_diacritics_all(descriptions)
    if components.remove_punctuation:
        remove_punctuation_all(descriptions)
    return descriptions


def convert_lower_all(descriptions):
    if isinstance(descriptions._descriptions[0].processed_text, list):
        fn = lambda txt: [i.lower() for i in txt]
    else:
        fn = lambda txt: txt.lower()
    descriptions.process_all(fn, "convert_lower")


def word_tokenize_all(descriptions, word_tokenizer=None, remove_stopwords=False):
    tokenizer_fn_name = word_tokenizer.__name__ if word_tokenizer is not None else "tokenize_text"
    word_tokenizer = word_tokenizer or (lambda *args, **kwargs: tokenize_text(*args, **kwargs)[1])
    if isinstance(descriptions._descriptions[0].processed_text, list):
        fn = lambda lst, stopwords: [word_tokenizer(i, stopwords) for i in lst]
    else:
        fn = lambda txt, stopwords: word_tokenizer(txt, stopwords)
    if remove_stopwords:
        indiv_kwargs = dict(stopwords=lambda desc: get_stopwords(desc.lang))
        tokenizer_fn_name += "_removestopwords"
    else:
        indiv_kwargs = dict(stopwords=lambda desc: None)
    descriptions.process_all(fn, tokenizer_fn_name, indiv_kwargs=indiv_kwargs, pgbar="Word-Tokenizing Descriptions")


#TODO use_better_german_tagger should be True!!
def lemmatize_all(descriptions, use_better_german_tagger=False):
    # see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung
    german_tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    lemmatizer = nltk.WordNetLemmatizer()
    assert "sent_tokenize" in descriptions.proc_steps
    assert not "removestopwords" in descriptions.proc_steps  # taggers work best on sentences
    def lemmatize(txt, language):
        lemmatized = []
        for sent in txt:
            if language == "de" and use_better_german_tagger:
                if isinstance(sent, list):
                    sent = " ".join(sent) #TODO really?!
                tags = german_tagger.tag_sent(sent)
                lemmatized.append([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
                raise NotImplementedError()
            else:
                tags = nltk.pos_tag(sent)
                lemmatized.append([lemmatizer.lemmatize(word, wntag(pos)) if wntag(pos) else word for word, pos in tags])
        return lemmatized
    descriptions.process_all(lemmatize, "lemmatize", indiv_kwargs=dict(language=lambda desc: desc.lang), pgbar="Lemmatizing Descriptions    ")


def remove_diacritics_all(descriptions):
    if isinstance(descriptions._descriptions[0].processed_text, list):
        if isinstance(descriptions._descriptions[0].processed_text[0], list):
            fn = lambda txt: [[unidecode.unidecode(word) for word in sent] for sent in txt]
        else:
            fn = lambda txt: [unidecode.unidecode(sent) for sent in txt]
    else:
        fn = unidecode.unidecode
    descriptions.process_all(fn, "remove_diacritics")


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
    if isinstance(descriptions._descriptions[0].processed_text, list):
        if isinstance(descriptions._descriptions[0].processed_text[0], list):
            return descriptions.process_all(remove_punctuation, "remove_punctuation")
    raise NotImplementedError()


# def make_bow(descriptions):
#     all_words = sorted(set(flatten([flatten(desc.processed_text) for desc in descriptions])))
#     for desc in descriptions:
#         if isinstance(desc.processed_text[0], str):
#             desc.bow = Counter(desc.processed_text)
#         else:
#             desc.bow = Counter(flatten(desc.processed_text))
#     return all_words, descriptions



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

def pmi(doc_term_matrix, positive=False, verbose=False, descriptions=None):
    # PMI as defined by DESC15
    logger.info("Calculating PMIs...")
    arr = doc_term_matrix.as_csr()
    total_words = arr.sum()
    arr = arr/total_words                 #now arr is p_{et}
    words_per_doc = arr.sum(axis=0)       #p_{e*}
    ges_occurs_per_term = arr.sum(axis=1) #p_{*t}
    prod = scipy.sparse.csr.csr_matrix(np.outer(ges_occurs_per_term, words_per_doc))
    res = arr/prod
    res = np.log1p(res)  # DESC15 say it's just the log, but if we take the log all the values 0<val<1 are negative and [i for i in res[doc_term_matrix.reverse_term_dict["building"]].tolist()[0] if i > 0] becomes a much smaller number
    if positive:
        res[res < 0] = 0.0
    quantifications  = csr_to_list(res.T)
    if verbose:
        print_quantification(doc_term_matrix, quantifications, descriptions)
    return quantifications

# def pmi(doc_term_matrix, positive=False, verbose=False, mds_obj=None, descriptions=None):
#     """
#     calculation of ppmi/pmi ([DESC15] 3.4 first lines)
#     see https://stackoverflow.com/a/58725695/5122790
#     see https://www.overleaf.com/project/609bbdd6a07c203c38a07ab4
#     """
#     logger.info("Calculating PMIs...")
#     arr = doc_term_matrix.as_csr()
#     #see doc_term_matrix.as_csr().toarray() - spalten pro doc und zeilen pro term
#     #[i for i in arr[doc_term_matrix.reverse_term_dict["building"], :].toarray()[0] if i > 0]
#     words_per_doc = arr.sum(axis=0)       #old name: col_totals
#     total_words = words_per_doc.sum()     #old name: total
#     ges_occurs_per_term = arr.sum(axis=1) #old name: row_totals
#     #assert np.array(ges_occurs_per_term.squeeze().tolist()).squeeze()[doc_term_matrix.reverse_term_dict["building"]] == np.array(ges_occurs_per_term.squeeze().tolist()).squeeze()[doc_term_matrix.reverse_term_dict["building"]]
#     expected = np.outer(ges_occurs_per_term, words_per_doc)
#     expected = np_divide(expected, total_words)  #TODO maybe I can convert this to a csr to save RAM?
#     quantifications = np_divide(arr, expected)
#     del expected
#     gc.collect()
#     # with np.errstate(divide='ignore'): # Silence distracting warnings about log(0)
#     quantifications = np_log(quantifications)
#     if positive:
#         quantifications[quantifications < 0] = 0.0
#     quantifications  = [[[i,elem] for i, elem in enumerate(quantifications[:,i]) if elem != 0] for i in tqdm(range(quantifications.shape[1]), desc="Last PPMI Step")]
#     if verbose:
#         print_quantification(doc_term_matrix, quantifications, descriptions)
#     return quantifications

ppmi = partial(pmi, positive=True)




# def pmi(arr, **kwargs):
#     '''
#     https://gist.github.com/TheLoneNut/208cd69bbca7cd7c53af26470581ec1e
#     Calculate the positive pointwise mutal information score for each entry
#     https://en.wikipedia.org/wiki/Pointwise_mutual_information
#     We use the log( p(y|x)/p(y) ), y being the column, x being the row
#     '''
#     # p(y|x) probability of each t1 overlap within the row
#     row_totals = arr.sum(axis=1).astype(float)
#     prob_cols_given_row = (arr.T / row_totals).T
#
#     # p(y) probability of each t1 in the total set
#     col_totals = arr.sum(axis=0).astype(float)
#     prob_of_cols = col_totals / sum(col_totals)
#
#     # PMI: log( p(y|x) / p(y) )
#     # This is the same data, normalized
#     ratio = prob_cols_given_row / prob_of_cols
#     ratio[ratio==0] = 0.00001
#     _pmi = np.log(ratio)
#     _pmi[_pmi < 0] = 0
#
#     return _pmi
#
# ppmi = pmi



#TODO use tf-idf as alternative keyword-detection! (erst mit gensim.dictionary alle Wörter holen, dann tf-idf drauffwerfen)
def tf_idf(doc_term_matrix, verbose=False, descriptions=None):
    """see https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016"""
    assert False, "Different result than sklearn!"
    n_docs = len(doc_term_matrix.dtm)
    quantifications = [[[term, count * log(n_docs/doc_term_matrix.doc_freqs[term])] for term, count in doc] for doc in doc_term_matrix.dtm]
    if verbose:
        print("Running TF-IDF on the corpus...")
        print_quantification(doc_term_matrix, quantifications, descriptions)
    return quantifications


def print_quantification(dtm, quantifications, descriptions):
    getname = lambda id: descriptions._descriptions[id].title
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



