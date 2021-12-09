import copy
import os
from itertools import accumulate
import six
from google.cloud import translate_v2 as translate
import html
import nltk
from itertools import accumulate

from derive_conceptualspace.settings import GOOGLE_CREDENTIALS_FILE

def translate_text(text, target="en", charlim=4900, origlans=None):
    # and I can still use the data from THIS call!!
    """Translates text into the target language. Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    Text can also be a sequence of strings, in which case this method will return a sequence of results for each text.
    """
    print(f"Translate-Charlim set to {charlim}")
    BYTELIM = int(204800*0.9) #if a request is bigger than google API will raise an Error!
    SEGLIM = 128 #https://github.com/googleapis/google-cloud-python/issues/5425#issuecomment-562745220
    TEXT_LEN_LIM = 2800 #google, this is getting ridiculus.
    SUMMED_TEXT_LEN_LIM = 100000 #102423 was too long...
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_FILE
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    translate_client = translate.Client()

    origtext = copy.deepcopy(text)
    if isinstance(text, (list, set, tuple)):
        accumulated_chars = list(accumulate([len(i) for i in text]))
        if accumulated_chars[-1] >= charlim:
            limit = [i > charlim for i in accumulated_chars].index(True)
            print(f"Have to drop {len(accumulated_chars)-limit} of {len(accumulated_chars)} elements!")
            if accumulated_chars[0] <= charlim:
                text = text[:limit-1 if limit >= 0 else None]
            else:
                print("Limit already reached!")
                return

    prelimtext = copy.deepcopy(text)
    res = []
    rest = []
    while True:
        while len("|".join(text).encode('utf-8')) > BYTELIM or len(text) > SEGLIM:
            rest.insert(0, text[-1])
            text = text[:-1]
        if any([len(i) > TEXT_LEN_LIM for i in text]):
            print("Fuck you google.")
            flatten = lambda l: [item for sublist in l for item in sublist]
            # splitfn = lambda txt, maxlen: [txt[i:i+maxlen] for i in range(0, len(txt)+maxlen, maxlen) if txt[i:i+maxlen]]
            # split_text = [i.split(". ") if len(i) > TEXT_LEN_LIM else [i] for i in text]
            split_text = [nltk.sent_tokenize(i) if len(i) > TEXT_LEN_LIM else [i] for i in text] #sent_tokenize(x, language=origlans[n]) but whatever
            assert all(len(i) <= TEXT_LEN_LIM for i in split_text)
            longer_index = {ind: len(elem) - 1 for ind, elem in enumerate(split_text) if len(elem) > 1}
            #now we merge the split sentences until they are all text-len-lim long
            for ind in longer_index.keys():
                lens = [len(i) for i in split_text[ind]]
                index_mapper = {0: 0}  # startindex: nwords
                indexmappernum = 0
                for num, elem in enumerate(lens):
                    assert elem <= TEXT_LEN_LIM, "one sentence is aleady too long."
                    if index_mapper[indexmappernum] + elem >= TEXT_LEN_LIM:
                        indexmappernum = num
                        index_mapper[indexmappernum] = 0
                    index_mapper[indexmappernum] += elem
                indices = list(index_mapper.keys()) + [len(split_text[ind]) + 1]
                indices = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
                split_text[ind] = ["".join(split_text[ind][i1:i2]) for i1, i2 in indices]
                longer_index[ind] = len(split_text[ind])-1
            text = [i[0] if isinstance(i, list) else i for i in split_text]
            latterparts = flatten([i[1:] for i in split_text if isinstance(i, list) and len(i) > 1])
            assert len(latterparts) <= SEGLIM, "fuck this."
            assert all(len(i) < TEXT_LEN_LIM for i in text)
            assert all(len(i) < TEXT_LEN_LIM for i in latterparts)
            assert sum([len(i) for i in text]) <= SUMMED_TEXT_LEN_LIM, "geez google what the actual fuck"
            assert sum([len(i) for i in latterparts]) <= SUMMED_TEXT_LEN_LIM, "geez google what the actual fuck"
            try:
                translations = translate_client.translate(text, target_language=target)
                translations2 = translate_client.translate(latterparts, target_language=target)
            except:
                failed = True
            else:
                failed = False
            assert sum(longer_index.values()) == len(translations2)
            latterparts_iter = iter(translations2)
            for index, ntranslations in longer_index.items():
                for i in range(ntranslations):
                    translations[index]["translatedText"] += next(latterparts_iter)["translatedText"]
            translated = translations
        else:
            assert all(len(i) < TEXT_LEN_LIM for i in text)
            try:
                translated = translate_client.translate(text, target_language=target)
            except:
                failed = True
            else:
                failed = False
        if not failed:
            assert len(translated) == len(text)
            res.extend(translated)
            assert len(res)+len(rest) == len(prelimtext)
            if rest:
                text = rest
                rest = []
            else:
                assert len(prelimtext) == len(res)
                break
        else:
            break

    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return [html.unescape(i["translatedText"]) for i in res]
