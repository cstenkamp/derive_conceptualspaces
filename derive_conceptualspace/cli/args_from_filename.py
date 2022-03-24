import argparse
import os
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import re

from dotenv.main import load_dotenv
from parse import parse
import ijson

from derive_conceptualspace.settings import standardize_config, ENV_PREFIX
from derive_conceptualspace.pipeline import generated_paths

LAST_RESULT = "cluster_reprs"

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', help='Filename you want to get envvars from')
    parser.add_argument('--variables', '-v', help='variables you want to get the filename from')
    return parser.parse_args()

def main():
    args = parse_command_line_args()
    if args.filename:
        print_envvars(args.filename)
    elif args.variables:
        get_filename(args.variables)
    else:
        inpt = input("What: ")
        if not " " in inpt:
            print_envvars(inpt)
        else:
            get_filename(inpt)


def apply_dotenv_vars(ENV_PREFIX="MA"):
    if os.getenv(ENV_PREFIX+"_SELECT_ENV_FILE"):
        assert isfile(os.getenv(ENV_PREFIX+"_SELECT_ENV_FILE"))
        load_dotenv(os.getenv(ENV_PREFIX+"_SELECT_ENV_FILE"))
    curr_envvars = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}
    curr_envvars = {k: os.path.expandvars(v) for k, v in curr_envvars.items()} #replace envvars
    curr_envvars = {k: os.path.expandvars(os.path.expandvars(re.sub(r"{([^-\s]*?)}", r"$\1", v))) for k, v in curr_envvars.items()} #replace {ENV_VAR} with $ENV_VAR and then apply them
    envfile = curr_envvars.get(ENV_PREFIX + "_" + "ENV_FILE")
    if envfile and not isfile(envfile) and isfile(join(curr_envvars.get(ENV_PREFIX+"_"+"CONFIGDIR", ""), envfile)):
        curr_envvars[ENV_PREFIX+"_"+"ENV_FILE"] = join(curr_envvars.get(ENV_PREFIX+"_"+"CONFIGDIR"), envfile)
    for k, v in curr_envvars.items():
        os.environ[k] = v



def get_filename(variables, get_dependencies=True, doprint=True):
    if not isinstance(variables, dict):
        variables = dict([[j.strip() for j in i.split(":")] for i in variables.split(",")])
    path = generated_paths[LAST_RESULT].format_map({k.lower(): v for k, v in variables.items()})
    if not get_dependencies:
        if doprint: print(path)
        return path
    else:
        if isfile(os.getenv("MA_SELECT_ENV_FILE", "")):
            load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
        apply_dotenv_vars()
        fname = join(os.getenv("MA_BASE_DIR"), path)
        with open(fname, "rb") as rfile:
            loadeds = next(ijson.items(rfile, "loaded_files"))
        dependencies = list({k:v["path"] for k, v in loadeds.items() if k not in ["raw_descriptions", "description_languages", "title_languages"]}.values())
        diff_dirs = [i for i in dependencies if dirname(dirname(path)) in i][0]
        pre_dir = diff_dirs[:diff_dirs.find(dirname(dirname(path)))]
        if doprint:
            print("Data-Dir:", pre_dir)
            print("Files:", ", ".join([i.replace(pre_dir, "") for i in dependencies]+[path]))
            print("Copy-command:", f"rsync -az --progress --relative grid:{' :'.join([i.replace(pre_dir, pre_dir+'.'+os.sep) for i in dependencies]+[pre_dir+'.'+os.sep+path])} "+os.environ["MA_BASE_DIR"])
            print("Env-vars:", get_envvars(path)[0])
        return path, pre_dir, dependencies

def get_envvars(filename):
    unmatched = []
    fname = splitext(basename(filename))[0]
    if re.match(r".*_[0-9]+", fname):
        fname = fname[:fname.rfind("_")]
    while fname not in generated_paths:
        unmatched.append(fname[fname.rfind("_"):])
        fname = fname[:fname.rfind("_")]
    pattern = generated_paths[fname]
    conf = parse(pattern, filename).named
    envvarstring = ";".join(f"{ENV_PREFIX}_{standardize_config(k,v)[0]}={standardize_config(k,v)[1]}" for k, v in conf.items())
    return envvarstring, fname

def print_envvars(filename):
    #TODO instead of envvars, maybe give full click-command (for that I'd need to know at which point stuff becomes relevant)
    envvarstring, fname = get_envvars(filename)
    print(envvarstring)
    print(fname)


if __name__ == '__main__':
    main()

