from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk.tree import Tree

from src.static.settings import DATA_BASE
from src.main.util.nltk_util import download_activate_stanfordnlp, download_nltk_dependencies

########################################################################################################################


# Defining a grammar & Parser
NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)

def get_continuous_chunks_a(text, chunk_func=ne_chunk):
    """https://stackoverflow.com/a/49584275/5122790"""
    download_nltk_dependencies()
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

def get_continuous_chunks_b(text, chunk_func=ne_chunk):
    """https://stackoverflow.com/a/49584275/5122790"""
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

########################################################################################################################
#https://taranjeet.co/nlp-detect-noun-phrase-and-verb-phrase/

def extract_phrase(tree_str, label):
    phrases = []
    trees = Tree.fromstring(tree_str)
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)
    return phrases

#TODO also verbrphrase
def stanford_extract_nounphrases(sentence):
    nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    tree_str = nlp.parse(sentence)
    nps = extract_phrase(tree_str, 'NP')
    return nps


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({'text':['This is a foo, bar sentence with New York city.',
                               'Another bar foo Washington DC thingy with Bruce Wayne.']})
    print(df['text'].apply(lambda sent: get_continuous_chunks_b((sent))))

