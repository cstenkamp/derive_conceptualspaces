import re
from tqdm import tqdm

from .get_candidates_keybert import WORD_NUM_APOSTR_REGEX
from derive_conceptualspace.util.tokenizers import phrase_in_text

flatten = lambda l: [item for sublist in l for item in sublist]

def fix_cand(cand, text):
    cand = cand.replace(" ’ s", "’s").replace("s ’", "s’")
    for char in ":,.);":
        cand = cand.replace(f" {char}", char)
    for char in "(":
        cand = cand.replace(f"{char} ", char)
    if cand not in text.lower():
        cand = cand.replace("‘ ", "‘")
        cand = cand.replace(": ", ":")
        cand = cand.replace(".:", " .: ")
        cand = cand.replace(" *", "*")
        cand = cand.replace(" '", "'")
    return cand


def postprocess_candidates(candidate_terms, descriptions):
    postprocessed_candidates = [[] for _ in candidate_terms]
    fails = set()
    for desc_ind, desc in enumerate(tqdm(descriptions)):
        for cand in candidate_terms[desc_ind]:
            if not phrase_in_text(cand, desc):
                cand = desc.split(" ")[desc[:desc.find(cand)].count(" "):][0]
                cand = "".join([i for i in cand if re.match(WORD_NUM_APOSTR_REGEX, i)])
                if not phrase_in_text(cand, desc):
                    fails.add(cand)
                    continue
            if cand.lower() not in desc.lower():
                if fix_cand(cand, desc).lower() in desc.lower() and phrase_in_text(fix_cand(cand, desc), desc):
                    cand = fix_cand(cand, desc)
                else:
                    fails.add(cand)
                    continue
            postprocessed_candidates[desc_ind].append(cand)
    for desc_ind, desc in enumerate(descriptions):
        for cand in postprocessed_candidates[desc_ind]:
            assert phrase_in_text(cand, desc)
            assert cand.lower() in desc.lower()
    print(f"Had to drop {len(fails)} out of {len(list(set(flatten(candidate_terms))))} candidates.")
    return postprocessed_candidates
