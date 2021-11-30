import re
from keybert import KeyBERT
from nltk.corpus import stopwords as nlstopwords

from src.main.create_spaces.text_tools import tokenize_text, phrase_in_text

WORD_REGEX = re.compile("[a-zA-ZäüöÜÄÖß-]+")        #TODO "[^\W\d_]" see https://stackoverflow.com/a/6314634/5122790 #TODO see https://stackoverflow.com/a/3617818/5122790
WORD_NUM_REGEX = re.compile("[a-zA-ZäüöÜÄÖß0-9-]+")
WORD_NUM_APOSTR_REGEX = re.compile("[a-zA-ZäüöÜÄÖß0-9'-]+")

class KeyBertExtractor():
    """https://github.com/MaartenGr/KeyBERT"""
    #TODO there really are many many configs and I think changing these changes a great deal! see https://github.com/MaartenGr/KeyBERT and try out stuff!!
    #TODO there is a minimum-frequency-argument!! https://github.com/MaartenGr/KeyBERT/blob/master/keybert/_model.py#L83-L101
    #TODO does this use the phrase_in_text function? SHOULD IT?

    stopwordlanguages = {"en": "english", "de": "german"}

    def __init__(self, is_multilan, faster=False):
        """available models: https://github.com/MaartenGr/KeyBERT#25-embedding-models"""
        assert not (is_multilan and faster)
        if faster:
            self.model_name = "paraphrase-MiniLM-L6-v2"
        elif is_multilan:
            self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        else:
            self.model_name = "paraphrase-mpnet-base-v2"
        print(f"Using model {self.model_name}")
        self.kw_model = KeyBERT(self.model_name)

    def _fix_hyphenated(self, cand, comparedtext):
        # it may be the case that the candiate is something like "particle systems", however the text only has "many-particle systems".
        # if so, then `(not phrase_in_text(cand, without_stops)) and cand in without_stops == True`
        words_before_onset = comparedtext[:comparedtext.find(cand)].count(" ")
        chars_before_onset = len(" ".join(comparedtext.split(" ")[:words_before_onset]))
        if chars_before_onset > 0 and chars_before_onset + 1 != comparedtext.find(cand):
            # then the first word is hyphenated
            return comparedtext[chars_before_onset + 1:comparedtext.find(cand) + len(cand)]
        elif words_before_onset == 0 and bool(re.fullmatch(WORD_NUM_REGEX, comparedtext[:comparedtext.find(cand)])):
            return comparedtext[:comparedtext.find(cand) + len(cand)]
        else:
            # then not the first word is hyphenated, but the last
            chars_after_hyphen = comparedtext[comparedtext.find(cand) + len(cand):].find(" ")
            if chars_after_hyphen > 0:
                return comparedtext[comparedtext.find(cand):comparedtext.find(cand) + len(cand) + chars_after_hyphen]
            elif re.fullmatch(WORD_NUM_REGEX, comparedtext[comparedtext.find(cand) + len(cand):]):
                return comparedtext[comparedtext.find(cand):]
        print("hm?!")
        return "NOPE"


    def extract_candidate(self, cand, text, without_stops, inds_without_stops, only_words, inds_only_words):
        #TODO not sure if this version can also correct hyphenated stuff like the old one ARGH!!

        if (not phrase_in_text(cand, without_stops)) and cand in without_stops:
            cand = self._fix_hyphenated(cand, without_stops)
            if phrase_in_text(cand, text): #maybe we're already done here
                return cand
            #now the cand is fixed and you can continue to checking phrase_in_text

        if phrase_in_text(cand, without_stops):
            tokenized_with_stops = tokenize_text(text, stopwords=None)[1]
            startpos = without_stops.find(cand)
            start_ind = without_stops[:startpos].count(" ")
            stoppos = startpos+len(cand)
            stop_ind = start_ind+without_stops[startpos:stoppos].count(" ")
            actual_phrase = " ".join(tokenized_with_stops[inds_without_stops[start_ind]:inds_without_stops[stop_ind]+1])
            if phrase_in_text(actual_phrase, text):
                if actual_phrase.split(" ")[0] == cand.split(" ")[0] and actual_phrase.split(" ")[-1] == cand.split(" ")[-1]:
                    # print(f"FROM {cand} TO {actual_phrase}")
                    return actual_phrase
                else:
                    print()
                    return
            print()
            return

        if (not phrase_in_text(cand, only_words)) and cand in only_words:
            cand = self._fix_hyphenated(cand, only_words)
            #now the cand is fixed and you can continue to checking phrase_in_text

        if phrase_in_text(cand, only_words):
            tokenized_with_stops = tokenize_text(text, stopwords=None)[1]
            startpos = only_words.find(cand)
            start_ind = only_words[:startpos].count(" ")
            stoppos = startpos+len(cand)
            stop_ind = start_ind+only_words[startpos:stoppos].count(" ")
            actual_phrase = " ".join(tokenized_with_stops[inds_only_words[start_ind]:inds_only_words[stop_ind]+1])
            if any(i in actual_phrase[:-1] for i in list("?!")+['"']): #if the phrase is not an actual phrase but split by punctuation
                print(f"{cand} is not an actual phrase - in the text it is `{actual_phrase}`")
                return None
            if phrase_in_text(actual_phrase, text):
                if actual_phrase.split(" ")[0] == cand.split(" ")[0] and actual_phrase.split(" ")[-1] == cand.split(" ")[-1]:
                    # print(f"FROM {cand} TO {actual_phrase}")
                    return actual_phrase
                else:
                    print()
                    return
            print()
            return

        if cand in without_stops:
            print("In without_stops")
            return

        if cand in only_words:
            print("in only_words")
            return

        #another thing: cand is "internship self organization", but in the text it's "internship self-organization". Maybe remove everything but letters and then re-apply?
        c2 = re.sub(re.compile(r'[\W\d]', re.U), "|", cand)
        t2 = re.sub(re.compile(r'[\W\d]', re.U), "|", text).lower()
        if c2 in t2:
            cand = text[t2.find(c2):t2.find(c2) + len(c2)]
            if phrase_in_text(cand, text):
                return cand
            else:
                print("whatever.")
        w2 = re.sub(re.compile(r'[\W\d]', re.U), "|", without_stops)
        if c2 in w2:
            cand = without_stops[w2.find(c2):w2.find(c2) + len(c2)]
            return self.extract_candidate(cand, text, without_stops, inds_without_stops, only_words, inds_only_words)
        o2 = re.sub(re.compile(r'[\W\d]', re.U), "|", only_words)
        if c2 in o2:
            cand = only_words[o2.find(c2):o2.find(c2) + len(c2)]
            return self.extract_candidate(cand, text, without_stops, inds_without_stops, only_words, inds_only_words)

        print(f"This does not work: {cand}")


    def __call__(self, text, lang="en"):
        """see scripts/notebooks/proof_of_concept/proofofconcept_keyBERT.ipynb for why this is like this"""
        assert lang in self.stopwordlanguages
        stopwords = tuple(nlstopwords.words(self.stopwordlanguages[lang]))

        candidates = set()
        for nwords in range(1, 4):
            n_candidates = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)

        #TODO: what if there are special chars in the candidates? is everything ok then with the word-splitting?
        #TODO does this work for numbers?!
        inds_without_stops, without_stops = tokenize_text(text, stopwords)
        ind_word_list = [(ind, word) for ind, word in zip(inds_without_stops, without_stops) if WORD_NUM_REGEX.fullmatch(word)]
        inds_only_words, only_words = list(zip(*ind_word_list)) if ind_word_list else ([], [])
        without_stops = " ".join(without_stops)
        only_words = " ".join(only_words)
        actual_keyphrases = []
        used_candidates = []
        n_immediateworking = n_fixed = n_errs = 0
        for cand in candidates:
            # if not all(WORD_REGEX.fullmatch(i) for i in cand.split(" ")):
            #     print(f"The candidate `{cand}` is not purely textual!")

            if phrase_in_text(cand, text):
                actual_keyphrases.append(cand)
                used_candidates.append(cand)
                n_immediateworking += 1
            else:
                intextcand = self.extract_candidate(cand, text, without_stops, inds_without_stops, only_words, inds_only_words)
                #TODO wenn in candidate ne zahl oder so ist die entfernen und es neu versuchen

                if intextcand:
                    if phrase_in_text(intextcand, text):
                        actual_keyphrases.append(intextcand)
                        used_candidates.append(cand)
                        n_fixed += 1
                        continue
                    else:
                        print("The extracted candidate is STILL not in the text!")
                n_errs += 1



        return actual_keyphrases, used_candidates, (n_immediateworking, n_fixed, n_errs)