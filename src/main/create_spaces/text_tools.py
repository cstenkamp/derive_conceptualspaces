from functools import lru_cache

from nltk.tokenize import word_tokenize


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
    inds, words = list(zip(*indwords))
    assert not any(" " in i for i in words)
    words = [char_replacer.get(i, i) for i in words]
    assert not any(" " in i for i in words)
    return inds, words


def phrase_in_text(phrase, text, return_count=False):
    #TODO ensure this is correct and all classes that should use this use this.
    text = tokenize_text(text)[1]
    text = " ".join(text).lower() #UPDATED 25.11.21 - I should re-do everything that uses this..
    text = " "+text+" "
    phrase = " "+phrase.lower()+" "
    if return_count:
        return text.count(phrase)
    return phrase in text