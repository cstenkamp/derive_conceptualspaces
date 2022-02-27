#! /home/student/c/cstenkamp/miniconda/envs/watcher/bin/python3

import re
import os
from os.path import join, isfile, basename
import time
import subprocess as sp
import shlex
import sys

from dotenv import load_dotenv
import yaml

LAST_READ = (0, None)

def load_acctfile():
    global LAST_READ
    if not os.path.isfile(os.environ["MA_CUSTOM_ACCTFILE"]):
        return {}
    if os.path.getmtime(os.getenv("MA_CUSTOM_ACCTFILE", join(os.getenv("HOME"), "custom_acctfile.yml"))) == LAST_READ[0]:
        return LAST_READ[1]
    for ntrial in range(1, 6):
        try:
            with open(os.getenv("MA_CUSTOM_ACCTFILE", join(os.getenv("HOME"), "custom_acctfile.yml")), "r") as rfile:
                custom_acct = yaml.load(rfile, Loader=yaml.SafeLoader)
            break
        except:
            time.sleep(ntrial)
    custom_acct = custom_acct if custom_acct is not None else {}
    LAST_READ = (os.path.getmtime(os.getenv("MA_CUSTOM_ACCTFILE", join(os.getenv("HOME"), "custom_acctfile.yml"))), custom_acct)
    return custom_acct


def get_active_jobs(acctfile=None):
    try:
        qstat_res = sp.check_output(shlex.split("qstat -s pr")).decode().strip()
    except FileNotFoundError:
        if acctfile is None: return []
        return list(acctfile.keys())
    res = {int(x.split()[0]) : x.split()[4] for x in qstat_res.splitlines()[2:]}
    return list(res.keys())


def get_enqueued_detailed():
    jobs = {}
    qstat_res = sp.check_output(shlex.split("qstat -r")).decode().strip()
    iterator = iter(qstat_res.split("\n"))
    # iterator = iter(txt.split("\n"))
    for line in iterator:
        var = ("snakejob" in line or "smk_runner" in line) and "cstenkamp" in line
        if var:
            id, prio, shortname, owner, status, date, time = [i for i in line.split(" ") if i][:7]
            namestring, name = [i.strip() for i in next(iterator).split(":")]; assert namestring == "Full jobname"
            jobs[id] = dict(name=name, status=status, date=date, time=time)
            if status == "r":
                machineshort, slots = [i for i in line.split(" ") if i][7:]
                queuestring, queue = [i.strip() for i in next(iterator).split(":")]; assert queuestring == "Master Queue"
                queue, machine = queue.split("@")
                jobs[id] = dict(**jobs[id], queue=queue, machine=machine, slots=slots)
    return jobs


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

########################################################################################################################
# stuff to work with error-files

def read_errorfile(error_file):
    # MA_BASE_DIR is set in the run_snakemake.sge file ($PWD and $MA_BASE_DIR are the same)
    error_file = join(os.environ["MA_BASE_DIR"], "..", "..", error_file) if isfile(join(os.environ["MA_BASE_DIR"], "..", "..", error_file)) else join(os.environ["MA_BASE_DIR"], error_file)
    if not isfile(error_file) and isfile(join(os.environ["MA_BASE_DIR"], basename(error_file)) if isfile(join(os.environ["MA_BASE_DIR"], basename(error_file))) else join(os.environ["MA_BASE_DIR"], "logs", "sge", basename(error_file))):
        error_file = join(os.environ["MA_BASE_DIR"], basename(error_file)) if isfile(join(os.environ["MA_BASE_DIR"], basename(error_file))) else join(os.environ["MA_BASE_DIR"], "logs", "sge", basename(error_file))
    try:
        with open(error_file, "r", newline="\n", errors="backslashreplace") as rfile:
            txt = rfile.read()
    except UnicodeDecodeError as e:
        print("Cannot read", error_file, file=sys.stderr)
        raise e
    return txt


def errorfile_from_id(jobid, acctfile):
    jobinfo = acctfile.get(str(jobid))
    if not jobinfo:
        return None, None
    if not ("envvars" in jobinfo and all (i in jobinfo["envvars"] for i in ["e", "rulename", "jobid"])):
        return None, None
    error_file = jobinfo["envvars"]["e"].format(rulename=jobinfo["envvars"]["rulename"], jobid=jobinfo["envvars"]["jobid"])
    txt = read_errorfile(error_file)
    return txt, error_file


def extract_error(txt):
    if isinstance(txt, str):
        txt = txt.split("\n")
    txt = [i.strip() for i in txt if i.strip()]
    startlines = [lnum for lnum, line in enumerate(txt) if line == "Building DAG of jobs..."]
    if startlines:
        txt = txt[max(startlines):]  # take only the last try into account
    if not 'Exiting because a job execution failed. Look above for error message' in txt:
        return False, "" #no error
    if any("Job killed after exceeding memory limits" in i for i in txt):
        return "Job killed after exceeding memory limits", "MemoryLimit"
    if not any(i.startswith(j) for i in txt for j in ["RuleException:", "SystemExit in line", "Interrupted at iteration"]):
        return "Unknown Exception", "Exception"
    if "RuleException:" in txt:
        txt = txt[txt.index("RuleException:")+1:]
        kind = "Exception"
    elif any(i.startswith("SystemExit in line") for i in txt):
        txt = txt[[i for i, elem in enumerate(txt) if elem.startswith("SystemExit in line")][0]:]
        kind = "SystemExit("+txt[1]+")"
    elif any(i.startswith("Interrupted at iteration") for i in txt):
        return "", "Interrupted"
    txt = txt[:txt.index('Exiting because a job execution failed. Look above for error message')]
    txt = [i for i in txt if not any(j in i for j in ["in __rule_", "telegram_notifier.py", "Shutting down, this might take some time."])]
    return "\n".join(txt), kind




if __name__ == '__main__':
    print(get_enqueued_detailed())