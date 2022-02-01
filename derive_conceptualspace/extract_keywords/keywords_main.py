from collections import Counter
from os.path import join, basename
import logging

from tqdm import tqdm
from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from .get_candidates_keybert import KeyBertExtractor
from .get_candidates_rules import extract_coursetype

from derive_conceptualspace.util.text_tools import tf_idf, get_stopwords, ppmi


from misc_util.pretty_print import pretty_print as print
from ..settings import get_setting
from ..util.dtm_object import DocTermMatrix, csr_to_list

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

def extract_candidateterms(pp_descriptions, extraction_method, max_ngram, verbose=False):
    if extraction_method == "keybert":
        raise NotImplementedError("TODO")
        # candidateterms, metainf = extract_candidateterms_keybert_nopp(pp_descriptions, max_ngram, faster_keybert, verbose=verbose)
    elif extraction_method == "pp_keybert":
        candidateterms, metainf = extract_candidateterms_keybert_preprocessed(pp_descriptions, max_ngram, get_setting("faster_keybert"), verbose=verbose)
    elif extraction_method == "tfidf":
        candidateterms, metainf = extract_candidateterms_quantific(pp_descriptions, max_ngram, quantific="tfidf", verbose=verbose)
    elif extraction_method == "ppmi":
        candidateterms, metainf = extract_candidateterms_quantific(pp_descriptions, max_ngram, quantific="ppmi", verbose=verbose)
    else:
        raise NotImplementedError()
    flattened = set(flatten(candidateterms))
    print("Terms I found: ", ", ".join([f"{k+1}-grams: {v}" for k, v in sorted(Counter([i.count(" ") for i in flattened]).items(), key=lambda x:x[0])]), "| sum:", len(flattened))
    metainf["n_candidateterms"] = len(flattened)
    return candidateterms, metainf

def extract_candidateterms_keybert_nopp(pp_descriptions, faster_keybert=False, verbose=False):
    vocab, descriptions = pp_descriptions.values()
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
    if verbose:
        print(f"Immediately working: {n_immediateworking_ges}")
        print(f"Fixed: {n_fixed_ges}")
        print(f"Errors: {n_errs_ges}")
    return candidateterms, {"keybertextractor_modelname": extractor.model_name}


def extract_candidateterms_keybert_preprocessed(descriptions, max_ngram, faster_keybert=False, verbose=False):
    from keybert import KeyBERT  # lazily loaded as it needs tensorflow which takes some time to init
    model_name = "paraphrase-MiniLM-L6-v2" if faster_keybert else "paraphrase-mpnet-base-v2"
    print(f"Using model {model_name}")
    candidateterms = []
    kw_model = KeyBERT(model_name)
    for desc in tqdm(descriptions._descriptions, desc="Running KeyBERT on descriptions"):
        stopwords = get_stopwords(desc.lang)
        candidates = set()
        for nwords in range(1, max_ngram):
            n_candidates = kw_model.extract_keywords(desc.processed_as_string(), keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)
        if (ct := extract_coursetype(desc)) and ct not in candidates:
            candidates += [ct]
        candidateterms.append(candidates)
    return candidateterms, {"keybertextractor_modelname": model_name, "kw_max_ngram": max_ngram}

#TODO play around with the parameters here!
def extract_candidateterms_quantific(descriptions, max_ngram, quantific, verbose=False):
    max_per_doc_abs = get_setting("QUANTEXTRACT_MAXPERDOC_ABS");  max_per_doc_rel = get_setting("QUANTEXTRACT_MAXPERDOC_REL")
    min_val = get_setting("QUANTEXTRACT_MINVAL"); min_val_percentile = get_setting("QUANTEXTRACT_MINVAL_PERC")
    min_per_doc = get_setting("QUANTEXTRACT_MINPERDOC"); forcetake_percentile = get_setting("QUANTEXTRACT_FORCETAKE_PERC")
    assert not (min_val and min_val_percentile)
    print(f"Loading Doc-Term-Matrix with min-term-count {get_setting('CANDIDATE_MIN_TERM_COUNT')}")
    dtm = descriptions.generate_DocTermMatrix(min_df=get_setting("CANDIDATE_MIN_TERM_COUNT"), max_ngram=max_ngram)
    #Now I'm filtering here, I originally didn't want to do that but it makes the processing incredibly much faster
    if quantific == "tfidf":
        # quant = tf_idf(dtm, verbose=verbose, descriptions=descriptions)
        quant = csr_to_list(TfidfTransformer().fit_transform(dtm.as_csr().T))
    elif quantific == "ppmi":
        quant = ppmi(dtm, verbose=verbose, descriptions=descriptions)
    else:
        raise NotImplementedError()
    metainf = dict()
    if min_val_percentile:
        min_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), min_val_percentile * 100)
        metainf.update(kw_min_val_percentile=min_val_percentile, kw_calculated_minval=min_val)
    else:
        metainf.update(kw_min_val=min_val)
    forcetake_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), forcetake_percentile * 100)
    candidates = [ [ sorted(i, key=lambda x:x[1], reverse=True),
                     min(round(len(i)*max_per_doc_rel), max_per_doc_abs),
                     set(j[0] for j in i if j[1] >= forcetake_val)]
                 for i in quant]
    all_candidates = [set([k[0] for k in i[0][:min_per_doc]])|set([j[0] for j in i[0] if j[1] >= min_val][:i[1]])|i[2] for i in candidates]
    if len(set(flatten(all_candidates))) == len(dtm.all_terms):
        metainf["all_terms_are_candidates"] = True
    return [[dtm.all_terms[j] for j in i] for i in all_candidates], metainf


########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_filtered_doc_cand_matrix(postprocessed_candidates, descriptions, min_term_count, dcm_quant_measure, use_n_docs_count=True, verbose=False):
    doc_cand_matrix = create_doc_cand_matrix(postprocessed_candidates, descriptions, verbose=verbose)
    return filter_keyphrases(doc_cand_matrix, descriptions, min_term_count=min_term_count, dcm_quant_measure=dcm_quant_measure, use_n_docs_count=use_n_docs_count, verbose=verbose)


def create_doc_cand_matrix(postprocessed_candidates, descriptions, verbose=False):
    assert len(postprocessed_candidates) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions._descriptions) for cand in postprocessed_candidates[ndesc])
    all_phrases = list(set(flatten(postprocessed_candidates)))
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    dictionary = corpora.Dictionary([all_phrases])
    dtm = [sorted([(nphrase, desc.count_phrase(phrase)) for nphrase, phrase in enumerate(all_phrases) if phrase in desc], key=lambda x:x[0]) for ndesc, desc in enumerate(tqdm(descriptions._descriptions, desc="Creating Doc-Cand-Matrix"))]
    #TODO statt dem ^ kann ich wieder SkLearn nehmen
    doc_term_matrix = DocTermMatrix(dtm=dtm, all_terms=all_phrases, verbose=verbose, quant_name="count")
    #TODO why do I even need to filter this uhm err
    if verbose:
        print("The 25 terms that are most often detected as candidate terms (incl. their #detections):",
              ", ".join(f"{k} ({v})" for k, v in sorted(dict(Counter(flatten(postprocessed_candidates))).items(), key=lambda x: x[1], reverse=True)[:25]))
    return doc_term_matrix


def filter_keyphrases(doc_cand_matrix, descriptions, min_term_count, dcm_quant_measure, verbose=False, use_n_docs_count=True):
    """name is a bit wrong, this first filters the doc-keyphrase-matrix and afterwards applies a quantification-measure on the dcm"""
    assert len(doc_cand_matrix.dtm) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions._descriptions) for cand in doc_cand_matrix.terms_per_doc()[ndesc])
    assert all(i[1] > 0 for doc in doc_cand_matrix.dtm for i in doc)
    filtered_dcm = DocTermMatrix.filter(doc_cand_matrix, min_count=min_term_count, use_n_docs_count=use_n_docs_count, verbose=verbose, descriptions=descriptions)
    #TODO: drop those documents without any keyphrase?!
    filtered_dcm = filtered_dcm.apply_quant(dcm_quant_measure)
    return filtered_dcm
