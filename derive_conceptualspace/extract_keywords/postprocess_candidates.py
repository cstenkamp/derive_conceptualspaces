from tqdm import tqdm
from itertools import product, permutations
from collections import Counter

from sklearn.feature_extraction.text import strip_accents_unicode

from derive_conceptualspace.settings import get_setting

flatten = lambda l: [item for sublist in l for item in sublist]

def extracted_literally():
    #some extraction-methods did extract literally, in which case I want to assert that no changes need to be done.
    return get_setting("EXTRACTION_METHOD") not in ["pp_keybert", "keybert"]

def postprocess_candidateterms(candidate_terms, descriptions, extraction_method):
    """
    In this method I'll try to fix candidate-terms and check if they are really in the descriptions they claim to be.
    To count the descriptions they are in, I'll both generate a new doc-term-matrix with the respective ngram
    AND check if it's in the literal text of the description, such that after this, i can savely forget the original descriptions and focus on DTMs.
    """
    if get_setting("DEBUG"):
        maxlen = min(len(candidate_terms), len(descriptions._descriptions), get_setting("DEBUG_N_ITEMS"))
        descriptions._descriptions = descriptions._descriptions[:maxlen]
        candidate_terms = candidate_terms[:maxlen]
    assert len(candidate_terms) == len(descriptions), f"Candidate Terms: {len(candidate_terms)}, Descriptions: {len(descriptions)}"
    flattened = set(flatten(candidate_terms))
    print("Extracted Unique Terms: ", ", ".join([f"{k+1}-grams: {v}" for k, v in sorted(Counter([i.count(" ") for i in flattened]).items(), key=lambda x:x[0])]), "| sum:", len(flattened))
    print("Most often extracted Terms:", ", ".join(f"{i[0]} ({i[1]} times)" for i in sorted(list(Counter(flatten(candidate_terms)).items()), key=lambda x:x[1], reverse=True)[:5]))
    max_found_ngram = max(i.count(" ") for i in flatten(candidate_terms))+1
    dtm = descriptions.generate_DocTermMatrix(min_df=1, max_ngram=max_found_ngram)[0] #TODO check if this works for all parameter-combis

    postprocessed_candidates = [[] for _ in candidate_terms]
    fails, changeds, toolong, ignores = set(), set(), set(), set()

    if extraction_method == "keybert":
        from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents, get_countvec
        assert PPComponents.from_str(descriptions.recover_settings["pp_components"]).use_skcountvec
        #this is my try to reproduce the preprocessing for the terms from keybert (as it said in some T0D0 somewhere) - TODO do the non-skcountvec-method as well!!
        cnt = get_countvec(**descriptions.recover_settings, max_ngram=1, min_df=1)
        processor = lambda cand: " ".join(cnt.build_analyzer()(cand))
        try_edit_fns = (processor, strip_accents_unicode, fix_cand, lambda x:x.lower()) #all PERMUTATIONS of these will be tried, that's a combinatorical explosion!
    else:
        try_edit_fns = ()
    all_edit_fns = flatten([list(permutations(try_edit_fns, i+1)) for i in range(len(try_edit_fns))])

    for desc_ind, desc in enumerate(tqdm(descriptions._descriptions, desc="Checking extracted candidates per-description")):
        term_counts = {dtm.all_terms[ind]: count for ind, count in dtm.dtm[desc_ind]}
        for cand in candidate_terms[desc_ind]:
            if cand.count(" ")+1 > (get_setting("MAX_NGRAM") or 1):
                toolong.add(cand)
                continue
            if "xxMA_SENTBORDERxx" in cand:
                ignores.add(cand)
                continue
            cond, ncand = check_cand(cand, desc, edit_fns=all_edit_fns)
            if cond:
                if not extracted_literally():
                    assert term_counts[ncand] == desc.count_phrase(ncand) #!!this shows that the DTM contains exactly the bow!!
                    if cand != ncand:
                        changeds.add((cand, ncand))
                postprocessed_candidates[desc_ind].append(ncand)
            else:
                fails.add(cand)

    if extracted_literally():
        assert not changeds and not toolong

    # changeds are for example when extract_coursetype extracted "seminar" from a description because it says "hauptseminar".
    # we can use that to make a mapping saying that a description containing the latter is defined to count as positive sample for the former.
    changeds_dict = {k: [] for k,vs in changeds}
    for k, v in changeds:
        changeds_dict[k].append(v)

    for desc_ind, desc in enumerate(tqdm(descriptions._descriptions, desc="Checking a second time "+("(quickly)" if descriptions.proc_steps == ["bow"] else "(slowly)"))):
        desc_txt = desc.processed_as_string(allow_shorten=True)
        desc_dtm = set(i[0] for i in dtm.dtm[desc_ind])
        for cand in postprocessed_candidates[desc_ind]:
            assert cand in desc
            if not descriptions.proc_steps == ["bow"]: #superflous check if it was created solely from the bow
                assert cand in desc_txt
            assert dtm.reverse_term_dict[cand] in desc_dtm

    if toolong:
        print(f"Had to drop {len(toolong)} out of {len(flatten(candidate_terms))} (non-unique) candidates because they were too long.")
    if ignores:
        print(f"Had to drop {len(ignores)} out of {len(flatten(candidate_terms))} (non-unique) candidates because they were across sentence borders.")
    print(f"Had to drop {len(fails)} out of {len(flatten(candidate_terms))} (non-unique) candidates"+(f" and edit {len(changeds)}." if changeds else "."))
    print("Postprocessed Unique Terms: ", ", ".join([f"{k+1}-grams: {v}" for k, v in sorted(Counter([i.count(" ") for i in set(flatten(postprocessed_candidates))]).items(), key=lambda x:x[0])]), "| sum:", len(set(flatten(postprocessed_candidates))))
    return postprocessed_candidates, changeds_dict


########################################################################################################################
# this stuff is only necessary for pp_keybert and keybert

def fix_cand(cand):
    cand = cand.replace(" ’ s", "’s").replace("s ’", "s’")
    for char in ":,.);":
        cand = cand.replace(f" {char}", char)
    for char in "(":
        cand = cand.replace(f"{char} ", char)
    cand = cand.replace("‘ ", "‘")
    cand = cand.replace(": ", ":")
    cand = cand.replace(".:", " .: ")
    cand = cand.replace(" *", "*")
    cand = cand.replace(" '", "'")
    return cand


def check_cand(cand, desc, edit_fns=None):
    if cand in desc:
        return True, cand

    if extracted_literally(): #TODO all of this (besides assert false) is dirty as argh!
        if ("lemmatize" in desc.list_ref.proc_steps and
                any("." in k for k in [j for j in flatten([i[0] for i in desc.processing_steps if i[1] == "lemmatize"][0]) if j.startswith(cand.split(" ")[0])])):
            # reason for this: In eg. "...  geschult.Der Kurs wird ...", due to no space after the dot and shitty sent-tokenization, "geschult.der" becomes a 1-gram ("geschult.d" after lemmatization).
            # the cand is extracted with sklearn with less shitty sent-tokenization, so it's "geschult kurs".
            return False, None
        if cand in flatten([i.split("-") for i in flatten(desc.processed_text) if "-" in i]):
            # reason for this: "sprachen" in the string "b-sprachen". Better would be to from-the-start if "b-sprachen" is in the text, add a bow-entry for "sprachen" as well
            # desc._bow[cand] = sum([desc.count_phrase(i) for i in flatten(desc.processed_text) if "-" in i and cand in i.split("-")])
            # if (ncand := [i for i in flatten(desc.processed_text) if "-" in i and any(j == cand for j in i.split("-"))][0]) in desc:
            #     return True, ncand
            return False, None
        if not max((indices := [flatten(desc.processed_text).index(i) if i in flatten(desc.processed_text) else -999 for i in cand.split(" ")]))-min(indices) <= cand.count(" ")+1:
            #see above, this time for situation: token='hausaufgabe geschult kurs', in text it is "hausaufgabe geschult d kurs" (geschult.der -> geschult.d -> geschult d, d being no stopword)
            return False, None
        # jo auf die ganzen anderen edge cases hab ich keinen Bock mehr. Dann sind die halt keine Candidates.
        # assert False, "You shouldn't get here with your extraction-method!"
        return False, cand

    part_cands = [[i for i in desc.bow().keys() if part in i] for part in cand.split(" ")]
    for ccand in product(*part_cands):
        if " ".join(ccand) in desc:
           return True, " ".join(ccand)

    for editors in (edit_fns or []):
        ncand = cand
        for fn in editors:
            ncand = fn(ncand)
        fixed_cond, fixed = check_cand(ncand, desc)
        if fixed_cond:
            return True, fixed

    return False, cand