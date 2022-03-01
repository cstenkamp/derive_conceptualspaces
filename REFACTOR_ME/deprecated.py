import os

def set_envvar(envvarname, value):
    if isinstance(value, bool):
        if value:
            os.environ[envvarname] = "1"
        else:
            os.environ[envvarname + "_FALSE"] = "1"
    else:
        os.environ[envvarname] = str(value)


def get_envvar(envvarname):
    if os.getenv(envvarname):
        tmp = os.environ[envvarname]
        if tmp.lower() == "none":
            return "none"
        elif tmp == "True":
            return True
        elif tmp == "False":
            return False
        elif tmp.isnumeric() and "DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):] in globals() and isinstance(globals()["DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):]], bool) and tmp in [0, 1, "0", "1"]:
            return bool(int(tmp))
        elif tmp.isnumeric():
            return int(tmp)
        elif all([i.isdecimal() or i in ".," for i in tmp]):
            return float(tmp)
        return tmp
    elif os.getenv(envvarname+"_FALSE"):
        return False
    return None



from functools import wraps
import sys

def notify_jsonpersister(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not kwargs.get("fordefault"):
            if hasattr(sys.stdout, "ctx") and "json_persister" in sys.stdout.ctx.obj:  # TODO getting the json_serializer this way is dirty as fuck!
                sys.stdout.ctx.obj["json_persister"].add_config(args[0].lower(), res)
        return res
    return wrapped



# @notify_jsonpersister
def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False, fordefault=True):
    #!!! diese funktion darf NICHTS machen außer sys.stdout.ctx.get_config(name) returnen!!! alles an processing gehört in die get_config!!!
    if hasattr(sys.stdout, "ctx"):
        return sys.stdout.ctx.get_config(name)
    #TODO einige Dinge von der old version waren schon sinnvoll, zum beispiel das bescheid sagen wenn von default, gucken
    # was ich davon wieder haben möchte
    # if fordefault: #fordefault is used for click's default-values. In those situations, it it should NOT notify the json-persister!
    #     silent = True
    #     stay_silent = False
    #     set_env_from_default = False
    #     default_none = True
    #     # return "default" #("default", globals().get("DEFAULT_"+name, "NO_DEFAULT"))
    # suppress_further = True if not silent else True if stay_silent else False
    # if get_envvar(get_envvarname(name, assert_hasdefault=False)) is not None:
    #     return get_envvar(get_envvarname(name, assert_hasdefault=False)) if get_envvar(get_envvarname(name, assert_hasdefault=False)) != "none" else None
    # if "DEFAULT_"+get_envvarname(name, assert_hasdefault=False, without_prefix=True) in globals():
    #     if not silent and not get_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup"):
    #         print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
    #     if suppress_further and not get_envvar(get_envvarname(name, assert_hasdefault=False) + "_shutup"):
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup", True)
    #     if set_env_from_default:
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+name, globals()['DEFAULT_'+name])
    #     return globals()["DEFAULT_"+name]
    # if default_none:
    #     return None
    # raise ValueError(f"There is no default-value for setting {name}, you have to explicitly pass it!")



def get_envvarname(config, assert_hasdefault=True, without_prefix=False):
    config = config.upper()
    if assert_hasdefault:
        assert "DEFAULT_"+config in globals(), f"there is no default value for {config}!"
    if without_prefix:
        return config
    return ENV_PREFIX+"_"+config

########################################################################################################################
########################################################################################################################
########################################################################################################################
#from json_persister:

    # def get_subdir(self, relevant_metainf, ignore_params=None):
    #     if not (self.dir_struct and all(i for i in self.dir_struct)):
    #         return "", []
    #     di = format_dict({**{k:v for k,v in self.ctx.obj.items() if k not in (ignore_params or [])}, **relevant_metainf})
    #     dirstruct = [d.format_map(di) for d in self.dir_struct]
    #     fulfilled_dirs = len(dirstruct) if not (tmp := [i for i, el in enumerate(dirstruct) if "UNDEFINED" in el]) else tmp[0]
    #     used_params = [k for k in di.keys() if "{"+k+"}" in "".join(self.dir_struct[:fulfilled_dirs])] #"verbrauchte", damit die nicht mehr zum filename hinzugefügt werden müssen
    #     return os.sep.join(dirstruct[:fulfilled_dirs]), used_params


    # def get_file_by_config(self, subdir, relevant_metainf, save_basename):
    #     subdirlen = len(join(self.in_dir, subdir))+1 if str(subdir).endswith(os.sep) else len(join(self.in_dir, subdir))
    #     candidates = [join(path, name)[subdirlen:] for path, subdirs, files in
    #                   os.walk(join(self.in_dir, subdir)) for name in files if name.startswith(save_basename)]
    #     candidates = [i if not i.startswith(os.sep) else i[1:] for i in candidates]
    #     assert candidates, f"No Candidate for {save_basename}! Subdir: {subdir}"
    #     if len(candidates) == 1:
    #         return candidates
    #     elif len(candidates) > 1:
    #         if all([splitext(i)[1] == ".json" for i in candidates]):
    #             correct_cands = []
    #             for cand in candidates:
    #                 tmp = json_load(join(self.in_dir, subdir, cand))
    #                 if (all(tmp.get("relevant_metainf", {}).get(k, v) == v or v == "ANY" for k, v in {**self.loaded_relevant_metainf, **relevant_metainf}.items()) and
    #                         # all(tmp.get("relevant_params", {}).get(k) for k, v in self.loaded_relevant_params.items()) and #TODO was this necessary
    #                         all(self.ctx.obj.get(k) == tmp["relevant_params"][k] for k in set(self.forward_params) & set(tmp.get("relevant_params", {}).keys()))):
    #                     correct_cands.append(cand)
    #             return correct_cands



def json_load(fname, **kwargs): #assert_meta=(), return_meta=False,
    if isinstance(fname, str):
        with open(fname, "r") as rfile:
            tmp = json.load(rfile, **kwargs)
    else: #then it may be a sacred opened resource (https://sacred.readthedocs.io/en/stable/apidoc.html#sacred.Experiment.open_resource)
        tmp = json.load(fname, **kwargs)
    # if isinstance(tmp, dict) and all(i in tmp for i in ["git_hash", "settings", "content"]):
    #     for i in assert_meta:
    #         assert getattr(settings, i) == tmp["settings"][i], f"The setting {i} does not correspond to what was saved!"
    #     if return_meta:
    #         meta = {k:v for k,v in tmp.items() if k != "content"}
    #         return npify_rek(tmp["content"]), meta
    #     return npify_rek(tmp["content"])
    return npify_rek(tmp)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# calculating first quants_s and from that the SVMs in create-candidate-svm-step:

# print(f"Starting Multiprocessed with {ncpu} CPUs")
# # with WorkerPool(get_ncpu(), dcm, pgbar="Counting Terms") as pool:
# #     quants_s = pool.work(terms, lambda dcm, term: dcm.term_quants(term))
# with SkipContext() as skipped, Interruptible(terms, [[], None, None], metainf, continue_from=continue_from,
#                                              contains_mp=True, name="Counting") as iter:
#     with WorkerPool(ncpu, dcm, pgbar="Counting Terms", comqu=iter.comqu) as pool:
#         quants_s, interrupted = pool.work(iter.iterable, lambda dcm, term: dcm.term_quants(term))
#     quants_s, _, _ = iter.notify([quants_s, None, None], exception=interrupted)
#     if interrupted is not False:
#         return quants_s, None, None, metainf
# if skipped.args is not None:
#     quants_s, _, _ = skipped.args
# assert len(quants_s) == len(terms)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# very old preprocessing func:

# def tokenize_sentences_nltk(descriptions):
#     #so as we're really only concerning bags-of-words here, we run a lemmatizer
#     # (see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung)
#     tagger = ht.HanoverTagger('morphmodel_ger.pgz')
#     res = []
#     words = set()
#     for n, sample in enumerate(tqdm(descriptions)):
#         all_tags = []
#         assert "sent_tokenize" in [i[1] for i in sample.processing_steps]
#         for sent in sample.processed_text:
#             tags = tagger.tag_sent(sent)
#             all_tags.extend([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
#         res.append(all_tags) # we could res.append(Counter(all_tags))
#         words.update(all_tags)
#     words = list(words)
#     alls = []
#     for wordlist in res:
#         cnt = Counter(wordlist)
#         alls.append(np.array([cnt[i] for i in words]))
#     return words, np.array(alls)


########################################################################################################################
########################################################################################################################
########################################################################################################################
# OLD PMI & TFIDF Calculation funcs:

# def pmi(doc_term_matrix, positive=False, verbose=False, mds_obj=None, descriptions=None):
#     """
#     calculation of ppmi/pmi ([DESC15] 3.4 first lines)
#     see https://stackoverflow.com/a/58725695/5122790
#     see https://www.overleaf.com/project/609bbdd6a07c203c38a07ab4
#     """
#     logger.info("Calculating PMIs...")
#     arr = doc_term_matrix.as_csr()
#     #see doc_term_matrix.as_csr().toarray() - spalten pro doc und zeilen pro term
#     #[i for i in arr[doc_term_matrix.reverse_term_dict["building"], :].toarray()[0] if i > 0]
#     words_per_doc = arr.sum(axis=0)       #old name: col_totals
#     total_words = words_per_doc.sum()     #old name: total
#     ges_occurs_per_term = arr.sum(axis=1) #old name: row_totals
#     #assert np.array(ges_occurs_per_term.squeeze().tolist()).squeeze()[doc_term_matrix.reverse_term_dict["building"]] == np.array(ges_occurs_per_term.squeeze().tolist()).squeeze()[doc_term_matrix.reverse_term_dict["building"]]
#     expected = np.outer(ges_occurs_per_term, words_per_doc)
#     expected = np_divide(expected, total_words)  #TODO maybe I can convert this to a csr to save RAM?
#     quantifications = np_divide(arr, expected)
#     del expected
#     gc.collect()
#     # with np.errstate(divide='ignore'): # Silence distracting warnings about log(0)
#     quantifications = np_log(quantifications)
#     if positive:
#         quantifications[quantifications < 0] = 0.0
#     quantifications  = [[[i,elem] for i, elem in enumerate(quantifications[:,i]) if elem != 0] for i in tqdm(range(quantifications.shape[1]), desc="Last PPMI Step")]
#     if verbose:
#         print_quantification(doc_term_matrix, quantifications, descriptions)
#     return quantifications


# def pmi(arr, **kwargs):
#     '''
#     https://gist.github.com/TheLoneNut/208cd69bbca7cd7c53af26470581ec1e
#     Calculate the positive pointwise mutal information score for each entry
#     https://en.wikipedia.org/wiki/Pointwise_mutual_information
#     We use the log( p(y|x)/p(y) ), y being the column, x being the row
#     '''
#     # p(y|x) probability of each t1 overlap within the row
#     row_totals = arr.sum(axis=1).astype(float)
#     prob_cols_given_row = (arr.T / row_totals).T
#
#     # p(y) probability of each t1 in the total set
#     col_totals = arr.sum(axis=0).astype(float)
#     prob_of_cols = col_totals / sum(col_totals)
#
#     # PMI: log( p(y|x) / p(y) )
#     # This is the same data, normalized
#     ratio = prob_cols_given_row / prob_of_cols
#     ratio[ratio==0] = 0.00001
#     _pmi = np.log(ratio)
#     _pmi[_pmi < 0] = 0
#
#     return _pmi
#
# ppmi = pmi


# def tf_idf(doc_term_matrix, verbose=False, descriptions=None):
#     """see https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016"""
#     assert False, "Different result than sklearn!"
#     n_docs = len(doc_term_matrix.dtm)
#     quantifications = [[[term, count * log(n_docs/doc_term_matrix.doc_freqs[term])] for term, count in doc] for doc in doc_term_matrix.dtm]
#     if verbose:
#         print("Running TF-IDF on the corpus...")
#         print_quantification(doc_term_matrix, quantifications, descriptions)
#     return quantifications