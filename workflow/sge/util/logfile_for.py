#! /home/student/c/cstenkamp/miniconda/envs/derive_conceptualspaces/bin/python3

import argparse
import os
from dotenv import load_dotenv
from os.path import isfile, join
import re

from sge_util import load_acctfile, get_active_jobs

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


def check_joblog(jobid, acctfile):
    jobinfo = acctfile.get(str(jobid))
    if not jobinfo:
        return None, None
    error_file = jobinfo["envvars"]["e"].format(rulename=jobinfo["envvars"]["rulename"], jobid=jobinfo["envvars"]["jobid"])
    error_file = join(os.environ["MA_BASE_DIR"], "..", "..", error_file) if isfile(join(os.environ["MA_BASE_DIR"], "..", "..", error_file)) else os.path.join(os.environ["MA_BASE_DIR"], error_file)
    # error_file = "/home/chris/deleteme/snakejob.create_candidate_svm.29.log"
    with open(error_file, "r", newline="\n") as rfile:
        txt = rfile.read()
    return txt, error_file


def init():
    if isfile(os.getenv("MA_SELECT_ENV_FILE", "")):
        load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
    apply_dotenv_vars()
    jobid = parse_command_line_args()
    return jobid


def main():
    jobid = init()
    acctfile = load_acctfile()
    if jobid:
        txt, path = check_joblog(jobid, acctfile)
        print("Filepath:", path)
        print("=="*50)
        print(txt)
        print()
        print("==" * 50)
    else:
        for jobid in get_active_jobs(acctfile):
            try:
                txt = check_joblog(jobid, acctfile)[0]
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
            elif 'Exiting because a job execution failed. Look above for error message' in final_lines:
                string += "\n   died!"
            else:
                try:
                    last_counter = [l for l in final_lines if "\r" in l][-1].split("\r")[-1]
                except IndexError:
                    string += "\n   currently running"
                else:
                    string += "\n   progress: "+last_counter
            print(string)



if __name__ == '__main__':
    main()