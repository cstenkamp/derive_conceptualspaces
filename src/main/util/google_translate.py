import copy
import os
from itertools import accumulate
import six
from google.cloud import translate_v2 as translate
import html

from src.static.settings import GOOGLE_CREDENTIALS_FILE

def translate_text(text, target="en", charlim=49000, origlans=None): #TODO charlim=490000
    # and I can still use the data from THIS call!!
    """Translates text into the target language. Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    Text can also be a sequence of strings, in which case this method will return a sequence of results for each text.
    """
    BYTELIM = int(204800*0.9) #if a request is bigger than google API will raise an Error!
    SEGLIM = 128 #https://github.com/googleapis/google-cloud-python/issues/5425#issuecomment-562745220
    TEXT_LEN_LIM = 3000 #google, this is getting ridiculus.
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
            splitfn = lambda txt, maxlen: [txt[i:i+maxlen] for i in range(0, len(txt)+maxlen, maxlen) if txt[i:i+maxlen]]
            # flatten = lambda l: [item for sublist in l for item in sublist]
            # split_text = [i.split(". ") if len(i) > TEXT_LEN_LIM else [i] for i in text]
            split_text = [splitfn(i, TEXT_LEN_LIM - 1) if len(i) > TEXT_LEN_LIM else [i] for i in text]
            assert all(len(i) <= TEXT_LEN_LIM for i in split_text)
            longer_index = {ind: len(elem) - 1 for ind, elem in enumerate(split_text) if len(elem) > 1}
            text = [i[0] if isinstance(i, list) else i for i in split_text]
            latterparts = flatten([i[1:] for i in split_text if isinstance(i, list) and len(i) > 1])
            #TODO die sätze einzelnd zu übersetzen macht die übersetzung definitiv schlechter.
            assert len(latterparts) <= SEGLIM, "fuck this."
            translations = translate_client.translate(text, target_language=target)
            translations2 = translate_client.translate(latterparts, target_language=target)
            assert sum(longer_index.values()) == len(translations2)
            latterparts_iter = iter(translations2)
            for index, ntranslations in longer_index.items():
                for i in range(ntranslations):
                    translations[index]["translatedText"] += ". " + next(latterparts_iter)["translatedText"]
            translated = translations
        else:
            translated = translate_client.translate(text, target_language=target)
        assert len(translated) == len(text)
        res.extend(translated)
        assert len(res)+len(rest) == len(prelimtext)
        if rest:
            text = rest
            rest = []
        else:
            break
    assert len(prelimtext) == len(res)


    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return [html.unescape(i["translatedText"]) for i in res]
