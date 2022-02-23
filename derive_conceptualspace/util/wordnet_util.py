from nltk.corpus import wordnet

#see https://github.com/Blubberli/germanetpy/blob/master/germanetpy_tutorial.ipynb, https://github.com/Germanet-sfs/germanetpy
from germanetpy.germanet import Germanet
from germanetpy.frames import Frames
from germanetpy.filterconfig import Filterconfig
from germanetpy.synset import WordCategory, WordClass


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


    germanet = Germanet("/home/chris/Documents/UNI_neu/Masterarbeit/data_new/germanet_v160/GN_V160_XML/")
    #TODO add path to respective env-files

    print()
    word = "Mathematik"
    synsets = germanet.get_synsets_by_orthform(word)
    print(f"{word} has {len(synsets)} senses:")
    for synset in synsets:
        print(f"  Synset id: {synset.id} | word category: {synset.word_category} | semantic field/wordclass: {synset.word_class}")
        print(f"    Direct Hypernyms: {synset.direct_hypernyms}")
        print(f"    Direct Hyponyms: {synset.direct_hyponyms}")
        for relation, synsets in synset.relations.items():
            print(f"      relation : {relation} | synsets: {synsets}")

    #TODO: write a function to get all hyponyms for a word from both wordnet and germanet, to be used for measuring the faithfulness of representation!