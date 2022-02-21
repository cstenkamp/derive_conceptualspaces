#! /home/student/c/cstenkamp/miniconda/envs/derive_conceptualspaces/bin/python3

import argparse
import os
import time
from os.path import isfile, join
import re
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from sge_util import load_acctfile, get_active_jobs, get_enqueued_detailed
from ikw_grid.sge_status import get_status

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
    try:
        with open(error_file, "r", newline="\n", errors="backslashreplace") as rfile:
            txt = rfile.read()
    except UnicodeDecodeError as e:
        print("Cannot read", error_file)
        raise e
    return txt, error_file


def init():
    if isfile(os.getenv("MA_SELECT_ENV_FILE", "")):
        load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
    apply_dotenv_vars()
    jobid = parse_command_line_args()
    return jobid

def print_singlejob(jobid, acctfile):
    txt, path = check_joblog(jobid, acctfile)
    print("Filepath:", path)
    print("==" * 50)
    print(txt)
    print()
    print("==" * 50)


def get_info_detailed(acctfile):
    new_info = {}
    for jobid, info in get_enqueued_detailed().items():
        age = datetime.now()-datetime.strptime(info["date"]+" "+info["time"], "%m/%d/%Y %H:%M:%S")
        info["age"] = str(age).split(".")[0]
        if info["name"] == "smk_runner":
            info["is_scheduler"] = True
        try:
            txt = check_joblog(jobid, acctfile)[0]
        except FileNotFoundError:
            # print("No log for job "+jobid+"! full info:", info)
            pass
        if txt:
            if len([line for line in txt.split("\n") if line.startswith("rule")]) == 1:
                rule_str = txt.split("\n")[[lnum for lnum, line in enumerate(txt.split("\n")) if line.startswith("rule")][0]:]
                info["rule"] = rule_str[0][len("rule "):-1]
                info["output"] = dict([i.strip().split(":") for i in rule_str[1:7] if len(i.split(":")) == 2])["output"].strip()
            final_lines = txt.split("\n")[-20:]
            if any("Saved under" in i for i in final_lines):
                saved_unter = [line for line in final_lines if "Saved under" in line][0][len("Saved under "):]
                info["saved_under"] = saved_unter.strip().rstrip(".")
                info["is_done"] = True
            elif 'Exiting because a job execution failed. Look above for error message' in final_lines:
                info["died"] = True
            else:
                try:
                    last_counter = [l for l in final_lines if "\r" in l][-1].split("\r")[-1]
                except IndexError:
                    info["progress"] = "running"
                else:
                    info["progress"] = last_counter
        new_info[jobid] = info
    return new_info


def print_multijob():
    infos = get_info_detailed(load_acctfile())
    while True:
        new_jobs = get_info_detailed(load_acctfile())
        infos.update(new_jobs)
        os.system("clear")

        schedulers = [(k,v) for k,v in new_jobs.items() if v.get("is_scheduler")]
        if len(schedulers) < 1:
            print("Scheduler died!")
        else:
            for schedulerid, schedulerinfo in schedulers:
                if not schedulerinfo.get("machine"):
                    print("Scheduler: Task-Id "+schedulerid+", scheduled for "+schedulerinfo["age"])
                else:
                    print("Scheduler: Task-Id "+schedulerid+", active for "+schedulerinfo["age"]+" on "+schedulerinfo["machine"].split(".")[0])

        active_jobs = {k: v for k, v in new_jobs.items() if v.get("status") == "r" and not v.get("is_scheduler")}
        if active_jobs: print("Active Jobs:")
        for jobid, info in active_jobs.items():
            print("  Job "+jobid+": does "+info.get("rule", info["name"].replace("snakejob.",""))+" for "+info["age"]+" on "+info["machine"].split(".")[0]+" with "+info["slots"]+" procs.")
            if "output" in info:
                print("      Creates "+info["output"])
            if "progress" in info:
                print("      Progress: "+info["progress"])

        scheduled_jobs = {k: v for k, v in new_jobs.items() if v.get("status") == "qw"}
        if scheduled_jobs: print("Scheduled Jobs:")
        for jobid, info in scheduled_jobs.items():
            print("  Job "+jobid+": will do "+info["name"].replace("snakejob.","")+". scheduled for "+info["age"])

        finished_jobs = {k: v for k, v in infos.items() if k not in new_jobs}
        for k in finished_jobs:
            if not finished_jobs[k].get("exit_stat"):
                finished_jobs[k]["exit_stat"] = get_status(k, silent=True)
        failed_jobs = {k: v for k, v in infos.items() if v.get("exit_stat") == "failed"}
        finished_jobs = {k: v for k, v in finished_jobs.items() if k not in failed_jobs}
        if finished_jobs: print("Finished Jobs:")
        for jobid, info in finished_jobs.items():
            try:
                print("  Job " + jobid + ": did " + info["rule"] + " at " + info["time"] + " on " +info["machine"].split(".")[0] + ", creating " + info["output"])
            except:
                print("  Job " + jobid + ": did " + info["name"].replace("snakejob.","") + " at " + info["time"])

        if failed_jobs: print("Failed Jobs:")
        for jobid, info in failed_jobs.items():
            print("  Job " + jobid + ": failed at " + info.get("rule", info["name"].replace("snakejob.","")))

        time.sleep(5)



def main():
    jobid = init()
    if jobid:
        acctfile = load_acctfile()
        print_singlejob(jobid, acctfile)
    else:
        print_multijob()



if __name__ == '__main__':
    main()