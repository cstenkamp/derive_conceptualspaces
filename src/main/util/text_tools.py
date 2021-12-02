from functools import lru_cache
from math import log

from nltk.tokenize import word_tokenize

flatten = lambda l: [item for sublist in l for item in sublist]

char_replacer = {
    '``': '"',
    '\'\'': '"',
    "“": '"',
    "”": '"',
}

@lru_cache(maxsize=None)
def tokenize_text(text, stopwords=None):
    text = text.lower()
    if stopwords:
        indwords = [(ind, word) for ind, word in enumerate(word_tokenize(text)) if word not in stopwords]
    else:
        indwords = [(ind, word) for ind, word in enumerate(word_tokenize(text))]
    inds, words = list(zip(*indwords)) if indwords else ([], [])
    assert not any(" " in i for i in words)
    words = [char_replacer.get(i, i) for i in words]
    assert not any(" " in i for i in words)
    return inds, words


def phrase_in_text(phrase, text, return_count=False):
    #TODO ensure this is correct and all classes that should use this use this.
    def process(txt, add_ending=False):
        if add_ending: txt = txt + " asdf" #why this? Because word_tokenize("dr.") != word_tokenize("dr. asdf")
        txt = tokenize_text(txt)[1]
        if add_ending: txt = txt[:-1]
        txt = " ".join(txt).lower()
        return " "+txt+" "
    text = process(text)
    phrase = process(phrase, add_ending=True)
    if return_count:
        return text.count(phrase)
    return phrase in text


#TODO use tf-idf as alternative keyword-detection! (erst mit gensim.dictionary alle Wörter holen, dann tf-idf drauffwerfen)
def tf_idf(doc_term_matrix, all_terms, verbose=False, mds_obj=None):
    """see https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016"""
    n_docs = len(doc_term_matrix)
    occurences = [set(i[0] for i in doc) for doc in doc_term_matrix]
    doc_freq = {term: sum(term in doc for doc in occurences) for term in tqdm(list(all_terms.keys()))}
    new_dtm = [[[term, count * log(n_docs/doc_freq[term])] for term, count in doc] for doc in doc_term_matrix]
    if verbose:
        print("Running TF-IDF on the corpus...")
        distinctive_terms = [max(doc, key=lambda x:x[1]) if doc else [-1, 0] for doc in new_dtm]
        most_distinct = sorted([[ind, elem] for ind, elem in enumerate(distinctive_terms)], key=lambda x:x[1][1], reverse=True)[:10]
        print("Most distinct terms:\n  "+"\n  ".join([f"`{all_terms[termid].ljust(12)}` ({str(round(value)).ljust(3)}) in `{mds_obj.names[docid]}`" for docid, (termid, value) in most_distinct]))
        frequent_words = [[all_terms[i[0]], i[1]] for i in sorted([[k, v] for k, v in doc_freq.items()], key=lambda x: x[1], reverse=True)[:20]]
        print("Terms that are in many documents:", ", ".join([f"{i[0]} ({round(i[1]/n_docs*100)}%)" for i in frequent_words]))
        values_per_phrase = {}
        for phrase, value in flatten(new_dtm):
            values_per_phrase.setdefault(phrase, []).append(value)
        average_phrase_val = {k: sum(v)/len(v) for k, v in values_per_phrase.items()}
        worst_phrases = [all_terms[i[0]] for i in sorted([[k,v] for k,v in average_phrase_val.items()], key=lambda x:x[1])[:20]]
        print("Keyphrases with the lowest average tf-idf score:", ", ".join(worst_phrases))
        #TODO maybe have an option to sort these out? But on the other hand, if half the courses contain the words `basic` or `introduction`, that's worth something
        #TODO alternatively also remove those terms with a high document frequency? important for LDA (aber dann brauch ich ne manuelle liste an to-keep (ich pack ja "seminar" explicitly rein))
    return new_dtm
