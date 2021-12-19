
#
# def display_mds(mds, names, max_elems=30):
#     """
#     Args:
#          mds: np.array or data_prep.jsonloadstore.Struct created from sklearn.manifold.MDS or sklearn.manifold.MSD
#          name: list of names
#          max_elems (int): how many to display
#     """
#     if hasattr(mds, "embedding_"):
#         mds = mds.embedding_
#     mins = np.argmin(np.ma.masked_equal(mds, 0.0, copy=False), axis=0)
#     for cmp1, cmp2 in enumerate(mins):
#         print(f"*b*{names[cmp1]}*b* is most similar to *b*{names[cmp2]}*b*")
#         if max_elems and cmp1 >= max_elems-1:
#             break
#
#
# def create_descstyle_dataset(n_dims, dsetname, from_path=SID_DATA_BASE, from_name_base="siddata_names_descriptions_mds_{n_dims}.json", to_path=SPACES_DATA_BASE, translate_policy=ORIGLAN):
#     names, descriptions, mds, languages = load_translate_mds(from_path, from_name_base.format(n_dims=n_dims), translate_policy)
#     display_mds(mds, names)
#     fname = join(to_path, dsetname, f"d{n_dims}", f"{dsetname}{n_dims}.mds")
#     os.makedirs(dirname(fname), exist_ok=True)
#     embedding = list(mds.embedding_)
#     indices = np.argsort(np.array(names))
#     names, descriptions, embedding = [names[i] for i in indices], [descriptions[i] for i in indices], np.array([embedding[i] for i in indices])
#     if isfile(namesfile := join(dirname(fname), "../main", "courseNames.txt")):
#         with open(namesfile, "r") as rfile:
#             assert [i.strip() for i in rfile.readlines()] == [i.strip() for i in names]
#     else:
#         with open(namesfile, "w") as wfile:
#             wfile.writelines("\n".join(names))
#     if isfile(fname):
#         raise FileExistsError(f"{fname} already exists!")
#     np.savetxt(fname, embedding, delimiter="\t")
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
# def load_translate_mds(file_path, file_name, translate_policy, assert_meta=(), translations_filename="translated_descriptions.json", assert_allexistent=True):
#     #TODO what now with this?! is this superflous? What about filgering the MDS?!
#     print("DEPRECATED!!!")
#     print(f"Working with file *b*{file_name}*b* in *b*{file_path}*b*!")
#     loaded = json_load(join(file_path, file_name), assert_meta=assert_meta)
#     names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]
#     if assert_allexistent:
#         assert len(names) == len(descriptions) == mds.embedding_.shape[0]
#     languages = create_load_languages_file(file_path, names, descriptions)
#     orig_n_samples = len(names)
#     additional_kwargs = {}
#     if translate_policy == ORIGLAN:
#         pass
#     elif translate_policy == ONLYENG:
#         indices = [ind for ind, elem in enumerate(languages) if elem == "en"]
#         print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones")
#         names, descriptions, languages = [names[i] for i in indices], [descriptions[i] for i in indices], [languages[i] for i in indices]
#         mds.embedding_ = np.array([mds.embedding_[i] for i in indices])
#         mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in indices])
#     elif translate_policy == TRANSL:
#         additional_kwargs["original_descriptions"] = descriptions
#         with open(join(file_path, translations_filename), "r") as rfile:
#             translations = json.load(rfile)
#         new_descriptions, new_indices = [], []
#         for ind, name in enumerate(names):
#             if languages[name] == "en":
#                 new_descriptions.append(descriptions[ind])
#                 new_indices.append(ind)
#             elif name in translations:
#                 new_descriptions.append(translations[name])
#                 new_indices.append(ind)
#         dropped_indices = set(range(len(new_indices))) - set(new_indices)
#         if dropped_indices:
#             print(f"Dropped {len(names) - len(new_indices)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
#         descriptions = new_descriptions
#         names, languages = [names[i] for i in new_indices], [list(languages.values())[i] for i in new_indices]
#         mds.embedding_ = np.array([mds.embedding_[i] for i in new_indices])
#         mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in new_indices])
#     descriptions = [html.unescape(i).replace("  ", " ") for i in descriptions]
#     if assert_allexistent:
#         assert len(names) == len(descriptions) == mds.embedding_.shape[0] == orig_n_samples
#     return MDSObject(names, descriptions, mds, languages, translate_policy, orig_n_samples, **additional_kwargs)

