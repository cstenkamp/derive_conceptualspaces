import copy
import os
from itertools import accumulate
import six
from google.cloud import translate_v2 as translate
import html

from src.static.settings import GOOGLE_CREDENTIALS_FILE

def translate_text(text, target="en", charlim=6500): #TODO check what happens if rate-limit reached,
    # and I can still use the data from THIS call!!
    """Translates text into the target language. Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    Text can also be a sequence of strings, in which case this method will return a sequence of results for each text.
    """
    BYTELIM = int(204800*0.9) #if a request is bigger than google API will raise an Error!
    SEGLIM = 128 #https://github.com/googleapis/google-cloud-python/issues/5425#issuecomment-562745220
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
