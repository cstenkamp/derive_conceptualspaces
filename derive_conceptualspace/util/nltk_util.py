from os.path import join, isdir, isfile
import nltk
import logging
import os
import urllib
import zipfile

# from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import wordnet as wn

from derive_conceptualspace.settings import get_setting

NLTK_LAN_TRANSLATOR = {
    "en": "english",
    "de": "german",
    "es": "spanish"
}

def download_nltk_dependencies(base_dir=None):
    kwargs = {}
    if base_dir:
        if nltk.data.path[0] != base_dir:
            nltk.data.path = [base_dir] + nltk.data.path
        kwargs['download_dir'] = base_dir
        os.makedirs(base_dir, exist_ok=True)

    try:
        _ = nltk.corpus.stopwords.words('german')
        _ = nltk.corpus.stopwords.words('english')
    except (LookupError, AttributeError, OSError) as e:
        logging.info("Downloading NLTK Stopwords")
        nltk.download('stopwords', **kwargs)

    try:
        _ = nltk.punkt
        nltk.word_tokenize('This is a text')
    except (LookupError, AttributeError, OSError) as e:
        logging.info("Downloading NLTK punkt")
        nltk.download('punkt', **kwargs)

    try:
        nltk.pos_tag(nltk.word_tokenize('This is a text'))
    except (LookupError, AttributeError, OSError) as e:
        logging.info("Downloading NLTK averaged_perceptron_tagger")
        nltk.download('averaged_perceptron_tagger', **kwargs)

    try:
        nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize('This is a text')))
    except (LookupError, AttributeError, OSError) as e:
        logging.info("Downloading NLTK maxent_ne_chunker")
        nltk.download('maxent_ne_chunker', **kwargs)
        logging.info("Downloading NLTK words")
        nltk.download('words', **kwargs)

    try:
        lemmatizer = nltk.WordNetLemmatizer()
        tags = nltk.pos_tag(nltk.word_tokenize('This is a text'))
        [lemmatizer.lemmatize(word, wntag(pos)) for word, pos in tags]
    except (LookupError, AttributeError, OSError) as e:
        logging.info("Downloading NLTK wordnet...")
        nltk.download('wordnet', **kwargs)


def download_activate_stanfordnlp(data_dir, langs):
    """https://stanfordnlp.github.io/CoreNLP/history.html"""
    #TODO - the model I'm using, is it cased or caseless? That's important!
    pkg = f"stanford-corenlp-{get_setting('STANFORDNLP_VERSION')}"
    if not (pkg in os.listdir(data_dir) and isdir(join(data_dir, pkg))):
        print(f"Downloading Stanfort Core NLP....")
        urllib.request.urlretrieve(f"https://nlp.stanford.edu/software/{pkg}.zip", filename=join(data_dir, pkg+".zip"))
        with zipfile.ZipFile(join(data_dir, pkg + ".zip")) as z:
            z.extractall(data_dir)
        os.remove(join(data_dir, pkg + ".zip"))
    if not isinstance(langs, (list, set, tuple)):
        langs = [langs]
    for lang in langs:
        version = f"stanford-corenlp-{STANFORDNLP_VERSION}-models-{lang}.jar"
        if not version in os.listdir(join(data_dir)):
            print(f"Downloading model {version}....")
            urllib.request.urlretrieve(f"http://nlp.stanford.edu/software/{version}", filename=join(data_dir, version))
        # TODO figure out alternative to symlinks for windows! maybe as context-manager that copies and removes after?
        # symlinked = f"stanford-corenlp-{STANFORDNLP_VERSION}-models.jar"
        # if isfile(join(data_dir, pkg, symlinked)):
        #     os.remove(join(data_dir, pkg, symlinked))
        if not isfile(join(data_dir, pkg, version)):
            os.symlink(join(data_dir, version), join(data_dir, pkg, version))
    os.environ["CLASSPATH"] = join(data_dir,pkg)
    nlp = StanfordCoreNLP(join(data_dir, pkg), logging_level=logging.INFO)
    return nlp


def wntag(pttag):
    """see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung"""
    if pttag in ['JJ', 'JJR', 'JJS']:
        return wn.ADJ
    elif pttag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wn.NOUN
    elif pttag in ['RB', 'RBR', 'RBS']:
        return wn.ADV
    elif pttag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wn.VERB
    return None