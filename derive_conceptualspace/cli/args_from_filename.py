import argparse
import os
from parse import parse
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import re

from derive_conceptualspace.settings import standardize_config, ENV_PREFIX
from derive_conceptualspace.pipeline import generated_paths

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Name of the file you want to auto-start')
    return parser.parse_args()

def main():
    args = parse_command_line_args()
    unmatched = []
    fname = splitext(basename(args.filename))[0]
    if re.match(r".*_[0-9]+", fname):
        fname = fname[:fname.rfind("_")]
    while fname not in generated_paths:
        unmatched.append(fname[fname.rfind("_"):])
        fname = fname[:fname.rfind("_")]
    pattern = generated_paths[fname]
    conf = parse(pattern, args.filename).named
    envvarstring = ";".join(f"{ENV_PREFIX}_{standardize_config(k,v)[0]}={standardize_config(k,v)[1]}" for k, v in conf.items())
    print(envvarstring)
    print(fname)

if __name__ == '__main__':
    main()

