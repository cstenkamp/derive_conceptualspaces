from nltk.tokenize import word_tokenize
from functools import lru_cache


char_replacer = {
    '``': '"',
    '\'\'': '"',
    "“": '"',
    "”": '"',
}

########################################################################################################################
########################################################################################################################
########################################################################################################################

# TODO these functions are only used for KeyBERT and should even there be replaced


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

########################################################################################################################
########################################################################################################################
########################################################################################################################

def phrase_in_text(phrase, text, return_count=False):
    # feb22: you can only use this in extracting candidates, not in postprocessing them!
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

