from collections import Counter
from os.path import join
import argparse
import pandas as pd
from misc_util.logutils import setup_logging
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.pipeline import SnakeContext, load_envfiles

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', help="Which classes should be targets for the dataset")
    return parser.parse_args()


def main():
    DATASET = "siddata"
    args = parse_command_line_args()
    setup_logging()
    load_envfiles(DATASET)

    ctx = SnakeContext.loader_context(silent=False)
    descriptions = ctx.load("pp_descriptions")
    res = extract_classes(descriptions, args.classes, ctx.obj["dataset_class"])
    fname = join(ctx.p.in_dir,  "fb_classifier", f"{DATASET}_{args.classes}.csv")
    res.reset_index().to_csv(fname)
    print(f"Saved under {fname}.")


def extract_classes(descriptions, classes, dataset_class, use_name=False):
    #TODO: merge this with the content of the very same thing in derive_conceptualspace.evaluate.shallow_trees.classify_shallowtree
    if classes is None:
        classes = descriptions.additionals_names[0]
    if classes in descriptions.additionals_names:
        catnames = None
        if hasattr(dataset_class, "CATNAMES") and classes in dataset_class.CATNAMES:
            catnames = dataset_class.CATNAMES.get(classes)
        hascat = [n for n, i in enumerate(descriptions._descriptions) if i._additionals[classes] is not None]
        getcat = lambda i: descriptions._descriptions[i]._additionals[classes]
    elif hasattr(dataset_class, "get_custom_class"):
        getcat, hascat, catnames = dataset_class.get_custom_class(classes, descriptions, verbose=True)
    else:
        raise Exception(f"The class {classes} does not exist!")
    if catnames and use_name:
        orig_getcat = getcat
        getcat = lambda x: catnames.get(int(orig_getcat(x)), orig_getcat(x))
    else:
        orig_getcat = getcat
        getcat = lambda x: int(orig_getcat(x))-1 #labels 0-9 instead of 1-10

    print(f"Using classes from {classes} - {len(hascat)}/{len(descriptions)} entities have a class")
    cats = {i: getcat(i) for i in hascat}
    print(f"Labels ({len(set(cats.values()))} classes):", ", ".join(f"*b*{k}*b*: {v}" for k, v in Counter(cats.values()).items()))
    return pd.DataFrame({descriptions._descriptions[i].title: [descriptions._descriptions[i].unprocessed_text, getcat(i)] for i in hascat}, index=["text", "class"]).T


if __name__ == '__main__':
    main()

#