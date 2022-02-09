from nltk.corpus import wordnet

def get_synonyms(term):
    return [l.name() for syn in wordnet.synsets(term) for l in syn.lemmas()]

def get_hypernyms(term, results=None, level=0):
    results = results or {}
    for syn in wordnet.synsets(term) if isinstance(term, str) else [term]:
        results.setdefault(level, []).append([lemma.name() for lemma in syn.lemmas()])
        for h in syn.hypernyms():
            get_hypernyms(h, results, level+1)
    return results

def get_hyponyms(term, results=None, level=0):
    results = results or {}
    for syn in wordnet.synsets(term) if isinstance(term, str) else [term]:
        results.setdefault(level, []).append([lemma.name() for lemma in syn.lemmas()])
        for h in syn.hyponyms():
            get_hyponyms(h, results, level+1)
    return results

def get_synset(term):
    return [i.name() for i in wordnet.synsets(term)]



if __name__ == '__main__':
    print(get_synset("math"))
    print(get_hyponyms("mathematics"))

    # print(get_hypernyms("mathematics"))
    # print(get_hypernyms("religion"))