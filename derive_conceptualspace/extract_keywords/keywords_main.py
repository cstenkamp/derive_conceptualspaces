from collections import Counter
from os.path import join, basename
import logging

from tqdm import tqdm
from gensim import corpora

from .postprocess_candidates import postprocess_candidates
from .get_candidates_keybert import KeyBertExtractor
from .get_candidates_rules import extract_coursetype

from derive_conceptualspace.util.text_tools import tf_idf, get_stopwords
from derive_conceptualspace.util.mpl_tools import show_hist
from derive_conceptualspace.util.jsonloadstore import json_load
from derive_conceptualspace.util.tokenizers import phrase_in_text, tokenize_text
from derive_conceptualspace.settings import CANDIDATETERM_MIN_OCCURSIN_DOCS

from misc_util.pretty_print import pretty_print as print
from ..util.dtm_object import DocTermMatrix

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

def extract_candidateterms_keybert(vocab, descriptions, faster_keybert=False):
    extractor = KeyBertExtractor(False, faster=faster_keybert)
    candidateterms = []
    n_immediateworking_ges, n_fixed_ges, n_errs_ges = 0, 0, 0
    for desc in tqdm(descriptions):
        keyberts, origextracts, (n_immediateworking, n_fixed, n_errs) = extractor(desc.text, desc.lang)
        #TODO maybe the best post-processing for all candidates is to run the exact processing I ran for the descriptions for all the results...?!
        # Because theoretically afterwards 100% of them should be in the processed_text
        if (ct := extract_coursetype(desc)) and ct not in keyberts:
            keyberts += [ct]
        candidateterms.append(keyberts)
        n_immediateworking_ges += n_immediateworking
        n_fixed_ges += n_fixed
        n_errs_ges += n_errs
    return candidateterms, extractor, (n_immediateworking_ges, n_fixed_ges, n_errs_ges)


def extract_candidateterms_keybert_preprocessed(vocab, descriptions, faster_keybert=False):
    from keybert import KeyBERT  # lazily loaded as it needs tensorflow which takes some time to init
    model_name = "paraphrase-MiniLM-L6-v2" if faster_keybert else "paraphrase-mpnet-base-v2"
    print(f"Using model {model_name}")
    candidateterms = []
    kw_model = KeyBERT(model_name)
    for desc in tqdm(descriptions):
        stopwords = get_stopwords(desc.lang)
        candidates = set()
        for nwords in range(1, 4):
            txt = ". ".join([" ".join(sent) for sent in desc.processed_text])
            n_candidates = kw_model.extract_keywords(txt, keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)
        if (ct := extract_coursetype(desc)) and ct not in candidates:
            candidates += [ct]
        candidateterms.append(candidates)
    return candidateterms, model_name


def postprocess_candidateterms(base_dir, descriptions):
    candidate_terms, meta_inf = json_load(join(base_dir, "candidate_terms.json"), return_meta=True)
    model = candidate_terms["model"]
    assert len(candidate_terms["candidate_terms"]) == len(descriptions), f"Candidate Terms: {len(candidate_terms['candidate_terms'])}, Descriptions: {len(descriptions)}"
    candidate_terms["candidate_terms"] = postprocess_candidates(candidate_terms, descriptions)
    # assert all(j.lower() in mds_obj.descriptions[i].lower() for i in range(len(mds_obj.descriptions)) for j in candidate_terms[i])
    return model, candidate_terms


########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_doc_cand_matrix(base_dir, descriptions, json_filename="candidate_terms_postprocessed.json", assert_postprocessed=True, verbose=False):
    candtermobj = json_load(join(base_dir, json_filename))
    assert not assert_postprocessed or candtermobj["postprocessed"]
    candidate_terms = candtermobj["candidate_terms"]
    assert len(candidate_terms) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions) for cand in candidate_terms[ndesc])
    all_phrases = list(set(flatten(candidate_terms)))
    # descriptions = [tokenize_text(i)[1] for i in mds_obj.descriptions]
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    dictionary = corpora.Dictionary([all_phrases])
    dtm = [sorted([(nphrase, desc.count_phrase(phrase)) for nphrase, phrase in enumerate(all_phrases) if phrase in desc], key=lambda x:x[0]) for ndesc, desc in enumerate(tqdm(descriptions))]
    doc_term_matrix = DocTermMatrix(all_phrases=all_phrases, dtm=dtm, descriptions=descriptions, verbose=verbose)
    if verbose:
        print("The 25 terms that are most often detected as candidate terms (incl. their #detections):",
              ", ".join(f"{k} ({v})" for k, v in sorted(dict(Counter(flatten(candidate_terms))).items(), key=lambda x: x[1], reverse=True)[:25]))
    return doc_term_matrix


def filter_keyphrases(base_dir, descriptions, min_term_count=10, matrix_val="count", cand_filename="candidate_terms_postprocessed.json",
                      dtm_filename="doc_cand_matrix.json", verbose=False, use_n_docs_count=True):
    assert matrix_val in ["count", "tf-idf", "binary"]
    candidate_terms = json_load(join(base_dir, cand_filename))["candidate_terms"]
    assert len(candidate_terms) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions) for cand in candidate_terms[ndesc])
    doc_term_matrix, all_terms = (tmp := json_load(join(base_dir, dtm_filename)))["doc_term_matrix"], tmp["all_terms"]
    if isinstance(list(all_terms.keys())[0], str):
        all_terms = {int(k): v for k,v in all_terms.items()}
    assert all(i[1] > 0 for doc in doc_term_matrix for i in doc)
    if use_n_docs_count:
        occurences = [set(i[0] for i in doc) for doc in doc_term_matrix]
        term_counts = {term: sum([term in i for i in occurences]) for term in all_terms.keys()}
    else:
        flat_terms = [flatten([[i[0]] * i[1] for i in doc]) for doc in doc_term_matrix]
        term_counts = Counter(flatten(flat_terms))
    used_terms = {k: v for k, v in term_counts.items() if v >= min_term_count}
    if verbose:
        print(f"Using only terms that occur "+(f"in at least {min_term_count} documents" if use_n_docs_count else f"at least {min_term_count} times")+f", which are {len(used_terms)} of {len(term_counts)} terms.")
        most_used = sorted(list(used_terms.items()), key=lambda x: x[1], reverse=True)[:10]
        print("The most used terms are: " + ", ".join([f"{all_terms[ind]} ({count})" for ind, count in most_used]))
        show_hist(list(used_terms.values()), "Occurences per Keyword", xlabel="Occurences per Keyword", cutoff_percentile=93)
    doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc] for doc in doc_term_matrix]
    all_terms = {all_terms[elem]: i for i, elem in enumerate(used_terms.keys())}; del used_terms
    doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc if ind in all_terms] for doc in doc_term_matrix]
    assert set(i[0] for doc in doc_term_matrix for i in doc) == set(all_terms.values())
    all_terms = {v: k for k, v in all_terms.items()}
    assert all(all_terms[ind] in descriptions[ndoc] for ndoc, doc in enumerate(tqdm(doc_term_matrix)) for ind, count in doc)
    if verbose:
        print("Documents without any keyphrase:", [descriptions[i] for i, e in enumerate(doc_term_matrix) if len(e) < 1][:5])
        print("Documents with just 1 keyphrase:", [[descriptions[i], all_terms[e[0][0]]] for i, e in enumerate(doc_term_matrix) if len(e) == 1][:5])
    #TODO: drop those documents without any keyphrase?!
    if matrix_val == "binary":
        doc_term_matrix = [[[ind, 1 if count >= 1 else 0] for ind, count in doc] for doc in doc_term_matrix]
    elif matrix_val == "tf-idf":
        doc_term_matrix = tf_idf(doc_term_matrix, all_terms, verbose=verbose, descriptions=descriptions)
    return doc_term_matrix, all_terms