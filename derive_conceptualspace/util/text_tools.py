import gc
import logging
from functools import partial, lru_cache
from math import log
from os.path import join, dirname, basename
import re

import numpy as np
import unidecode
from HanTa import HanoverTagger as ht
from nltk.corpus import stopwords as nlstopwords
from nltk import sent_tokenize as nltk_sent_tokenize, WordNetLemmatizer as nltk_WordNetLemmatizer, pos_tag as nltk_pos_tag
from tqdm import tqdm
import scipy.sparse.csr
from sklearn.feature_extraction.text import strip_accents_unicode

from derive_conceptualspace.settings import get_setting
from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.util.dtm_object import csr_to_list
from derive_conceptualspace.util.nltk_util import NLTK_LAN_TRANSLATOR, wntag
from derive_conceptualspace.util.np_tools import np_divide, np_log
from derive_conceptualspace.util.tokenizers import tokenize_text
from misc_util.pretty_print import pretty_print as print

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
def get_stopwords(language, include_desc15_stopwords=True, include_custom=True, include_withoutdiacritics=True):
    if language in NLTK_LAN_TRANSLATOR:
        language = NLTK_LAN_TRANSLATOR[language]
    assert language in NLTK_LAN_TRANSLATOR.values(), f"Cannot deal with language {language}"
    stopwords = set(nlstopwords.words(language))
    if include_desc15_stopwords and language == "english":
        stopwords |= load_desc15_stopwords()
    if include_custom and language == "english":
        stopwords |= set(get_setting("CUSTOM_STOPWORDS"))
    if include_withoutdiacritics:
        stopwords |= set(strip_accents_unicode(i) for i in stopwords)
    return tuple(stopwords)

################## TODO move these into a preprocessing main file ####################

def run_preprocessing_funcs(descriptions:DescriptionList, components, word_tokenizer=None):
    #components: sent_tokenize=True, lemmatize=True, remove_stopwords=True, convert_lower=True, remove_diacritcs=True, remove_punctuation=True
    # TODO use TreeTagger? https://textmining.wp.hs-hannover.de/Preprocessing.html#Alternative:-Treetagger
    # https://textmining.wp.hs-hannover.de/Preprocessing.html#Satzerkennung-und-Tokenization
    assert components.convert_lower, "Stopwords are lower-case so not converting is not allowed (bad for german...!)"
    if components.remove_htmltags:
        descriptions.process_all(lambda data: re.compile(r'<.*?>').sub('', data), "remove_htmltags")
    if components.sent_tokenize:
        if not get_setting("USE_STANZA"):
            descriptions.process_all(nltk_sent_tokenize, "sent_tokenize", indiv_kwargs=dict(language=lambda desc: NLTK_LAN_TRANSLATOR[desc.lang])) #nltk suuucks!! sent_tokenize(*, language=german) trennt sogar "...am Ende des 2. Semesters", oder, even worse, "Relevante Probleme wie z.B. Lautierungsregeln", but if there's no space after a dot it DOESN'T split obvious sentences! very visible in description "!! F??LLT AB 15.11. AUS !! Lekt??rekurs Spanisch I (Gruppe A und B)." #TODO!
            #TODO maybe write small rule-based-after-thingy that handles common stuff like... * "z.B." wird nicht getrennt * "\d+\. Nomen" (bspw "2. Semester") wird nicht getrennt * Kram wie "15.11." (also daten) wird nicht getrennt, ...
        else:
            logging.getLogger('stanza').setLevel(logging.ERROR)
            import stanza
            if len(descriptions.languages) > 1: raise NotImplementedError()
            nlp = stanza.Pipeline(lang='de', processors='tokenize')
            fn = lambda txt: [i._text for i in nlp(txt).sentences]
            descriptions.process_all(fn, "sent_tokenize", pgbar="Stanza Sentence-Tokenizing")
    if components.convert_lower:
        convert_lower_all(descriptions)
    #tokenization will happen anyway!
    if not components.lemmatize:
        word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=components.remove_stopwords)
    else:
        word_tokenize_all(descriptions, word_tokenizer=word_tokenizer, remove_stopwords=False)
        lemmatize_all(descriptions, components.convert_lower, components.remove_punctuation)
        if components.remove_stopwords:
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


class Lemmatizer():
    def __init__(self, descriptions, convert_lower, remove_punctuation):
        self.convert_lower = convert_lower
        self.remove_punctuation = remove_punctuation
        self.all_languages = descriptions.languages
        assert "sent_tokenize" in descriptions.proc_steps
        assert not "removestopwords" in descriptions.proc_steps  # taggers work best on sentences
        if "de" in descriptions.languages:
            self.german_tagger = ht.HanoverTagger('morphmodel_ger.pgz') # see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung
        if "en" in descriptions.languages:
            self.english_lemmatizer = nltk_WordNetLemmatizer()

    def __call__(self, txt, language=None):
        if language is None:
            assert len(self.all_languages) == 1
            language = list(self.all_languages)[0]
        lemmatized = []
        for sent in txt:
            if language == "de":
                assert isinstance(sent, list)
                tags = self.german_tagger.tag_sent(sent)
                fn = lambda x: x.casefold() if self.convert_lower else lambda x: x
                if self.remove_punctuation:
                    lemmatized.append([fn(i[1]) for i in tags if i[1] != "--"])
                else:
                    lemmatized.append([fn(i[1]) if i[1] != "--" else i[0] for i in tags])
            elif language == "en":
                tags = nltk_pos_tag(sent)
                lemmatized.append([self.english_lemmatizer.lemmatize(word, wntag(pos)) if wntag(pos) else word for word, pos in tags])
        return [i for i in lemmatized if i]


def lemmatize_all(descriptions, convert_lower, remove_punctuation):
    lemmatizer = Lemmatizer(descriptions, convert_lower, remove_punctuation)
    if len(descriptions.languages) > 1: raise NotImplementedError("Would be implemented for non-multilan")
    # descriptions.process_all(lemmatizer, "lemmatize", pgbar="Lemmatizing Descriptions", indiv_kwargs=dict(language=lambda desc: desc.lang))
    descriptions.process_all(lemmatizer, "lemmatize", pgbar="Lemmatizing Descriptions", multiproc=True)


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
                            #     print(f"Error at {word}") #TODO irgendwann muss ich mich darum k??mmern
        sents.append(words)
    return sents


def remove_punctuation_all(descriptions):
    if isinstance(descriptions._descriptions[0].processed_text, list):
        if isinstance(descriptions._descriptions[0].processed_text[0], list):
            return descriptions.process_all(remove_punctuation, "remove_punctuation")
    raise NotImplementedError()

#################################### END move these into a preprocessing main file #####################################
########################################################################################################################
########################################################################################################################

#TODO move PMI somewhere else as well

def pmi(doc_term_matrix, positive=False, verbose=False, descriptions=None):
    # PMI as defined by DESC15
    logger.info("Calculating PMIs...")
    arr = doc_term_matrix.as_csr()
    total_words = arr.sum()
    arr = arr/total_words                 #now arr is p_{et}
    words_per_doc = arr.sum(axis=0)       #p_{e*}
    ges_occurs_per_term = arr.sum(axis=1) #p_{*t}
    prod = ges_occurs_per_term*words_per_doc #I'd like to scipy.sparse.csr.csr_matrix(...), but that conversion kills my RAM completely..
    res = arr/prod
    res[np.isnan(res)] = 0
    del arr; del prod; gc.collect()
    res = np.log1p(res)  # DESC15 say it's just the log, but if we take the log all the values 0<val<1 are negative and [i for i in res[doc_term_matrix.reverse_term_dict["building"]].tolist()[0] if i > 0] becomes a much smaller number
    if positive:
        res[res < 0] = 0.0
    assert not np.isnan(res).any(), "There are NaNs in the PPMI!"
    quantifications  = csr_to_list(res.T)
    del res; gc.collect()
    if verbose:
        print("The counting that'll come now will take long and is only there because you're verbose")
        print_quantification(doc_term_matrix, quantifications, descriptions)
    return quantifications


ppmi = partial(pmi, positive=True)


########################################################################################################################


def print_quantification(dtm, quantifications, descriptions):
    getname = lambda id: descriptions._descriptions[id].repr(50)
    distinctive_terms = [max(doc, key=lambda x:x[1]) if doc else [-1, 0] for doc in quantifications]
    most_distinct = sorted([[ind, elem] for ind, elem in enumerate(distinctive_terms)], key=lambda x:x[1][1], reverse=True)[:10]
    print("Most distinct terms:\n  "+"\n  ".join([f"*r*{dtm.all_terms[termid].ljust(12)}*r* ({str(round(value)).ljust(3)}) in `{getname(docid)}`" for docid, (termid, value) in most_distinct]))
    frequent_words = [[dtm.all_terms[i[0]], i[1]] for i in sorted([[k, v] for k, v in dtm.term_freqs().items()], key=lambda x: x[1], reverse=True)[:20]]
    print("Terms that are in many documents:", ", ".join([f"{i[0]} ({round(i[1]/len(dtm.dtm)*100)}%)" for i in frequent_words]))
    values_per_phrase = {}
    for phrase, value in flatten(quantifications):
        values_per_phrase.setdefault(phrase, []).append(value)
    average_phrase_val = {k: sum(v)/len(v) for k, v in values_per_phrase.items()}
    worst_phrases = [dtm.all_terms[i[0]] for i in sorted([[k,v] for k,v in average_phrase_val.items()], key=lambda x:x[1])[:20]]
    print("Keyphrases with the lowest average score:", ", ".join(worst_phrases))
    #TODO maybe have an option to sort these out? But on the other hand, if half the courses contain the words `basic` or `introduction`, that's worth something
    #TODO alternatively also remove those terms with a high document frequency? important for LDA (aber dann brauch ich ne manuelle liste an to-keep (ich pack ja "seminar" explicitly rein))

