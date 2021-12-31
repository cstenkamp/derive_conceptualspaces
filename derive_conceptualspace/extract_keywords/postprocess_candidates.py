from tqdm import tqdm
from itertools import product

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


def postprocess_candidateterms(candidate_terms, descriptions, extraction_method):
    candidate_terms, = candidate_terms.values()
    _, descriptions = descriptions.values()
    assert len(candidate_terms) == len(descriptions), f"Candidate Terms: {len(candidate_terms)}, Descriptions: {len(descriptions)}"

    postprocessed_candidates = [[] for _ in candidate_terms]
    fails = set()
    in_text = (lambda phrase, desc: phrase in desc) if extraction_method == "pp_keybert" else (lambda phrase, desc: phrase_in_text(phrase, desc.text))

    for desc_ind, desc in enumerate(tqdm(descriptions)):
        for cand in candidate_terms[desc_ind]:
            cond, ncand = check_cand(cand, desc, in_text)
            if cond:
                postprocessed_candidates[desc_ind].append(ncand)
            else:
                fails.add(cand)

    for desc_ind, desc in enumerate(descriptions):
        for cand in postprocessed_candidates[desc_ind]:
            assert in_text(cand, desc)
            assert cand.lower() in desc.processed_as_string()
    print(f"Had to drop {len(fails)} out of {len(list(set(flatten(candidate_terms))))} candidates.")
    return postprocessed_candidates



def check_cand(cand, desc, in_text, try_fixing=True):
    if in_text(cand, desc):
        return True, cand

    # cand = desc.split(" ")[desc[:desc.find(cand)].count(" "):][0]
    # cand = "".join([i for i in cand if re.match(WORD_NUM_APOSTR_REGEX, i)])
    part_cands = [[i for i in desc.bow.keys() if part in i] for part in cand.split(" ")]
    for ccand in product(*part_cands):
        if in_text(" ".join(ccand), desc):
           return True, " ".join(ccand)

    if try_fixing: #this variable is recursive anchor
        fixed_cond, fixed = check_cand(fix_cand(cand, desc.processed_as_string()), desc, in_text, try_fixing=False)
        if fixed_cond:
            #TODO check if this is ever reached
            return True, fixed

    return False, cand