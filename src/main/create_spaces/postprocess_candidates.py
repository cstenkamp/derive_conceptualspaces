import re
from tqdm import tqdm

from src.main.create_spaces.get_candidates_keybert import WORD_NUM_APOSTR_REGEX
from src.main.create_spaces.text_tools import phrase_in_text

flatten = lambda l: [item for sublist in l for item in sublist]

def fix_cand(cand, text):
    cand = cand.replace(" ’ s", "’s").replace("s ’", "s’")
    for char in ":,.);":
        cand = cand.replace(f" {char}", char)
    for char in "(":
        cand = cand.replace(f"{char} ", char)
    if cand not in text:
        cand = cand.replace("‘ ", "‘")
        cand = cand.replace(": ", ":")
        cand = cand.replace(".:", " .: ")
        cand = cand.replace(" *", "*")
        cand = cand.replace(" '", "'")
    if cand not in text:
        cand = cand.replace("region is compared to the 'other ' mediterranean", "region is compared to the 'other' mediterranean")
    return cand

def postprocess_candidates(candidate_terms, descriptions):
    postprocessed_candidates = [[] for _ in candidate_terms]
    fails = set()
    for desc_ind, desc in enumerate(tqdm(descriptions)):
        desc = desc.replace("  ", " ").lower()
        for cand in candidate_terms[desc_ind]:
            orig_cand = cand
            if not phrase_in_text(cand, desc):
                cand = desc.split(" ")[desc[:desc.find(cand)].count(" "):][0]
                cand = "".join([i for i in cand if re.match(WORD_NUM_APOSTR_REGEX, i)])
                if not phrase_in_text(cand, desc):
                    fails.add(cand)
                    continue
            if cand.lower() not in desc:
                if fix_cand(cand, desc).lower() in desc and phrase_in_text(fix_cand(cand, desc), desc):
                    cand = fix_cand(cand, desc)
                else:
                    fails.add(cand)
                    continue
            postprocessed_candidates[desc_ind].append(cand)
    for desc_ind, desc in enumerate(descriptions):
        for cand in postprocessed_candidates[desc_ind]:
            assert phrase_in_text(cand, desc.replace("  ", " "))
            assert cand.lower() in desc.replace("  ", " ").lower()
    print(f"Had to drop {len(fails)} out of {len(list(set(flatten(candidate_terms))))} candidates.")
    return postprocessed_candidates
