from collections import Counter
from os.path import join, basename
import logging

from tqdm import tqdm
from gensim import corpora

from .postprocess_candidates import postprocess_candidates
from .get_candidates_keybert import KeyBertExtractor
from .get_candidates_rules import extract_coursetype

from derive_conceptualspace.util.text_tools import tf_idf
from derive_conceptualspace.util.mpl_tools import show_hist
from derive_conceptualspace.util.jsonloadstore import json_load
from derive_conceptualspace.util.tokenizers import phrase_in_text, tokenize_text
from derive_conceptualspace.settings import CANDIDATETERM_MIN_OCCURSIN_DOCS

from misc_util.pretty_print import pretty_print as print

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

def extract_candidateterms_keybert(mds_obj):
    extractor = KeyBertExtractor(False, faster=False)
    candidateterms = []
    n_immediateworking_ges, n_fixed_ges, n_errs_ges = 0, 0, 0
    for desc in tqdm(mds_obj.descriptions):
        keyberts, origextracts, (n_immediateworking, n_fixed, n_errs) = extractor(desc)
        if (ct := extract_coursetype(desc)) and ct not in keyberts:
            keyberts += [ct]
        candidateterms.append(keyberts)
        n_immediateworking_ges += n_immediateworking
        n_fixed_ges += n_fixed
        n_errs_ges += n_errs
    return candidateterms, extractor, (n_immediateworking_ges, n_fixed_ges, n_errs_ges)


def postprocess_candidateterms(base_dir, mds_obj):
    candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    model = candidate_terms["model"]
    candidate_terms = candidate_terms["candidate_terms"]
    assert len(candidate_terms) == len(mds_obj.descriptions)
    candidate_terms = postprocess_candidates(candidate_terms, mds_obj.descriptions)
    assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    return model, candidate_terms


########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_doc_term_matrix(base_dir, mds_obj, json_filename="candidate_terms_postprocessed.json", assert_postprocessed=True, verbose=False):
    candidate_terms = json_load(join(base_dir, json_filename))
    assert not assert_postprocessed or candidate_terms["postprocessed"]
    candidate_terms = candidate_terms["candidate_terms"]
    assert len(candidate_terms) == len(mds_obj.descriptions)
    assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    all_terms = list(set(flatten(candidate_terms)))
    descriptions = [tokenize_text(i)[1] for i in mds_obj.descriptions]
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    dictionary = corpora.Dictionary([all_terms])
    doc_term_matrix = [sorted([(ind, phrase_in_text(elem, mds_obj.descriptions[j], return_count=True)) for ind,elem in enumerate(all_terms) if phrase_in_text(elem, mds_obj.descriptions[j])], key=lambda x:x[0]) for j in tqdm(range(len(mds_obj.descriptions)))]
    if verbose:
        #TODO what's done here is now part of the dtm-object class, so take it from there?
        occurs_in = [set(j[0] for j in i) if i else [] for i in doc_term_matrix]
        num_occurences = [sum([term_ind in i for i in occurs_in]) for term_ind in tqdm(range(len(all_terms)))]
        show_hist(num_occurences, "Docs per Keyword", xlabel="# Documents the Keyword appears in", ylabel="Count (log scale)", cutoff_percentile=97, log=True)
        above_threshold = len([i for i in num_occurences if i>= CANDIDATETERM_MIN_OCCURSIN_DOCS])
        sorted_canditerms = sorted([[ind, elem] for ind, elem in enumerate(num_occurences)], key=lambda x:x[1], reverse=True)
        print(f"Found {len(all_terms)} candidate Terms, {above_threshold} ({round(above_threshold/len(all_terms)*100)}%) of which occur in at least {CANDIDATETERM_MIN_OCCURSIN_DOCS} descriptions.")
        print("The 25 terms that are most often detected as candidate terms (incl. their #detections):",
              ", ".join(f"{k} ({v})" for k, v in sorted(dict(Counter(flatten(candidate_terms))).items(), key=lambda x: x[1], reverse=True)[:25]))
        print("The 25 terms that occur in the most descriptions (incl the #descriptions they occur in):",
              ", ".join([f"{all_terms[ind]} ({occs})" for ind, occs in sorted_canditerms[:25]]))

    return all_terms, doc_term_matrix


def filter_keyphrases(base_dir, mds_obj, min_term_count=10, matrix_val="count", json_filename="candidate_terms_postprocessed.json", verbose=False):
    assert matrix_val in ["count", "tf-idf", "binary"]
    candidate_terms = json_load(join(base_dir, json_filename))["candidate_terms"]
    assert len(candidate_terms) == len(mds_obj.descriptions)
    assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    doc_term_matrix, all_terms = (tmp := json_load(join(base_dir, "doc_term_matrix.json")))["doc_term_matrix"], tmp["all_terms"]
    assert all(i[1] > 0 for doc in doc_term_matrix for i in doc)
    flat_terms = [flatten([[i[0]] * i[1] for i in doc]) for doc in doc_term_matrix]
    term_counts = Counter(flatten(flat_terms))
    used_terms = {k: v for k, v in term_counts.items() if v >= min_term_count}
    if verbose:
        print(f"Using only terms that occur at least {min_term_count} times, which are {len(used_terms)} of {len(term_counts)} terms.")
        most_used = sorted(list(used_terms.items()), key=lambda x: x[1], reverse=True)[:10]
        print("The most used terms are: " + ", ".join([f"{all_terms[ind]} ({count})" for ind, count in most_used]))
        show_hist(list(used_terms.values()), "Occurences per Keyword", xlabel="Occurences per Keyword", cutoff_percentile=93)
    doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc] for doc in doc_term_matrix]
    all_terms = {all_terms[elem]: i for i, elem in enumerate(used_terms.keys())}; del used_terms
    doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc if ind in all_terms] for doc in doc_term_matrix]
    assert set(i[0] for doc in doc_term_matrix for i in doc) == set(all_terms.values())
    all_terms = {v: k for k, v in all_terms.items()}
    assert all(phrase_in_text(all_terms[ind], mds_obj.descriptions[ndoc]) for ndoc, doc in enumerate(tqdm(doc_term_matrix)) for ind, count in doc)
    if verbose:
        print("Documents without any keyphrase:", [mds_obj.descriptions[i] for i, e in enumerate(doc_term_matrix) if len(e) < 1][:5])
        print("Documents with just 1 keyphrase:", [[mds_obj.descriptions[i], all_terms[e[0][0]]] for i, e in enumerate(doc_term_matrix) if len(e) == 1][:5])
    #TODO: drop those documents without any keyphrase?!
    if matrix_val == "binary":
        doc_term_matrix = [[[ind, 1 if count >= 1 else 0] for ind, count in doc] for doc in doc_term_matrix]
    elif matrix_val == "tf-idf":
        doc_term_matrix = tf_idf(doc_term_matrix, all_terms, verbose=verbose, mds_obj=mds_obj)
    return doc_term_matrix, all_terms