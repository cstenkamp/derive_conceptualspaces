#! /home/student/c/cstenkamp/miniconda/envs/derive_conceptualspaces/bin/python3

import yaml
import argparse
import os
from dotenv import load_dotenv
from os.path import isfile, join
import re
import subprocess as sp
import shlex

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

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', '-j')
    return parser.parse_args().jobid

def check_joblog(jobid):
    with open(os.getenv("MA_CUSTOM_ACCTFILE", join(os.getenv("HOME"), "custom_acctfile.yml")), "r") as rfile:
        custom_acct = yaml.load(rfile, Loader=yaml.SafeLoader)
    jobinfo = custom_acct.get(str(jobid))
    if not jobinfo:
        return None
    error_file = jobinfo["envvars"]["e"].format(rulename=jobinfo["envvars"]["rulename"], jobid=jobinfo["envvars"]["jobid"])
    error_file = os.path.join(os.environ["MA_BASE_DIR"], error_file)
    # error_file = "/home/chris/deleteme/snakejob.preprocess_descriptions_notranslate.2.log"
    with open(error_file, "r", newline="\n") as rfile:
        txt = rfile.read()
    return txt


def init():
    if isfile(os.getenv("MA_SELECT_ENV_FILE", "")):
        load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
    apply_dotenv_vars()
    jobid = parse_command_line_args()
    return jobid


def main():
    jobid = init()
    if jobid:
        txt = check_joblog(jobid)
        print("=="*50)
        print(txt)
        print()
        print("==" * 50)
    else:
        for jobid in get_active_jobs():
            try:
                txt = check_joblog(jobid)
            except FileNotFoundError:
                print("Job " + str(jobid) + ": not started")
                continue
            if not txt:
                continue
            string = "Job " + str(jobid) + ": "
            if len([line for line in txt.split("\n") if line.startswith("rule")]) == 1:
                rule_str = txt.split("\n")[[lnum for lnum, line in enumerate(txt.split("\n")) if line.startswith("rule")][0]:]
                rulename = rule_str[0][len("rule "):-1]
                output = dict([i.strip().split(":") for i in rule_str[1:7] if len(i.split(":")) == 2])["output"].strip()
                string += "does "+rulename+" for output "+output
            final_lines = txt.split("\n")[-20:]
            if any("Saved under" in i for i in final_lines):
                saved_unter = [line for line in final_lines if "Saved under" in line][0][len("Saved under "):]
                saved_under = saved_unter.strip().rstrip(".")
                string += "\n   is done and saved under "+saved_under
            else:
                last_counter = [l for l in final_lines if "\r" in l][-1].split("\r")[-1]
                string += "\n   progress: "+last_counter
            print(string)

def get_active_jobs():
    qstat_res = sp.check_output(shlex.split("qstat -s pr")).decode().strip()
    res = {int(x.split()[0]) : x.split()[4] for x in qstat_res.splitlines()[2:]}
    return list(res.keys())


if __name__ == '__main__':
    main()