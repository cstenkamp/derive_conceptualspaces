from keybert import KeyBERT
from nltk.corpus import stopwords as nlstopwords
from nltk.tokenize import word_tokenize


class KeyBertExtractor():
    """https://github.com/MaartenGr/KeyBERT"""
    #TODO there really are many many configs and I think changing these changes a great deal! see https://github.com/MaartenGr/KeyBERT and try out stuff!!
    #TODO there is a minimum-frequency-argument!! https://github.com/MaartenGr/KeyBERT/blob/master/keybert/_model.py#L83-L101

    stopwordlanguages = {"en": "english", "de": "german"}

    def __init__(self, is_multilan, faster=False):
        """available models: https://github.com/MaartenGr/KeyBERT#25-embedding-models"""
        assert not (is_multilan and faster)
        if faster:
            self.kw_model = KeyBERT("paraphrase-MiniLM-L6-v2")
        elif is_multilan:
            self.kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
        else:
            self.kw_model = KeyBERT("paraphrase-mpnet-base-v2")

    def __call__(self, text, lang="en"):
        """see scripts/notebooks/proof_of_concept/proofofconcept_keyBERT.ipynb for why this is like this"""
        assert lang in self.stopwordlanguages
        stopwords = nlstopwords.words(self.stopwordlanguages[lang])

        candidates = set()
        for nwords in range(1, 4):
            n_candidates = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)

        indwords = [(ind,word) for ind,word in enumerate(word_tokenize(text)) if word not in stopwords]
        inds, words = list(zip(*indwords))
        assert not any(" " in i for i in words)
        withoutstops = " ".join(words).lower()
        start_positions = [((start := withoutstops.find(i)), start+len(i)) for i in candidates]
        start_indices_withoutstops = [withoutstops[:i].count(" ") for i,j in start_positions]
        actual_keyphrases = []
        for start_ind, (startpos, stoppos) in zip(start_indices_withoutstops, start_positions):
            if start_ind >= 0 and startpos >= 0:
                full_phrase = withoutstops[startpos:stoppos]
                last_word = full_phrase.split(" ")[-1]
                from_start = word_tokenize(text.lower())[inds[start_ind]:]
                try:
                    actual_phrase = from_start[:from_start.index(last_word)+1]
                except:
                    continue
                if len(actual_phrase) < 10:
                    if " ".join(actual_phrase) in text.lower():
                        actual_keyphrases.append(" ".join(actual_phrase))
        return actual_keyphrases, candidates