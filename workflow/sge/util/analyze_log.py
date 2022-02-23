#! /home/student/c/cstenkamp/miniconda/envs/derive_conceptualspaces/bin/python3

import os
import argparse
from os.path import join, isfile
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv.main import load_dotenv

from logfile_for import print_singlejob, load_acctfile, check_joblog, extract_error, read_errorfile, apply_dotenv_vars
from ikw_grid.sge_status import get_status

flatten = lambda l: [item for sublist in l for item in sublist]

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('start_pid', help='ID of the first smk_runner that startet it all')
    parser.add_argument('--base_dir', '-b', help='Base-dir where the smk_runner logs are', default=os.environ["HOME"])
    parser.add_argument('--include_errors', '-e', help="If you want the tracebacks to be shown", default=False, action="store_true")
    return parser.parse_args()

def read_files(pid, base_dir):
    with open(join(base_dir, f"smk_runner.e{pid}")) as rfile:
        errorfile = rfile.read()
    with open(join(base_dir, f"smk_runner.o{pid}")) as rfile:
        outputfile = rfile.read()
    return errorfile, outputfile

def get_all_jobfiles(new_job_id, base_dir):
    errorfiles, outputfiles = [], []
    for _ in range(100):
        errorfile, outputfile = read_files(new_job_id, base_dir)
        assert outputfile.split("\n")[0] == f"This is run nr {len(errorfiles)+1}"
        errorfiles.append(errorfile)
        outputfiles.append(outputfile)
        if any(l.startswith("Your job") and l.endswith("has been submitted") for l in outputfile.split("\n")):
            line = [l for l in outputfile.split("\n") if l.startswith("Your job") and l.endswith("has been submitted")][0]
            new_job_id = line[len("Your job "):line[len("Your job "):].find(" ") + len("Your job ")]
        else:
            break
    for i, err in enumerate(errorfiles):
        if "Will exit after finishing currently running jobs." in err.split("\n"):
            errorfiles[i] = "\n".join(err.split("\n")[:err.split("\n").index("Will exit after finishing currently running jobs.")])
    return errorfiles, outputfiles


def split_all(errorfiles):
    todo_jobs = ["\n".join(i.split("\n")[i.split("\n").index("Job stats:")+1:i.split("\n").index("Select jobs to execute...")]) for i in errorfiles]

    job_strings = [i for i in flatten([i.split("\n")[i.split("\n").index("Select jobs to execute...")+1:] for i in errorfiles]) if i]
    job_strings = [i for i in job_strings if i != "Select jobs to execute..."]
    done_inds = [n for n, i in enumerate(job_strings) if i.startswith("Finished job")]
    dones = ["\n".join(job_strings[i - 1:i + 2]) for i in done_inds]
    for done in dones: job_strings = "\n".join(job_strings).replace(done, "").split("\n")
    dones = {i.split("\n")[1][len("Finished job "):-1]:datetime.strptime(i.split("\n")[0][1:-1], "%a %b %d %H:%M:%S %Y") for i in dones}
    job_strings = [i for i in job_strings if i]

    start_inds = [n for n, i in enumerate(job_strings) if i.startswith("rule") and i.endswith(":")]
    indiv_rules = [[j.strip() for j in job_strings[start_inds[i]-1:start_inds[i+1]-1]] for i in range(len(start_inds)-1)]
    job_infos = []
    leftover_text = []
    for nrule, rule in enumerate(indiv_rules):
        job_infos.append(dict([j.split(": ") for j in rule[2:[n for n, j in enumerate(rule) if j.startswith("Will submit") or j.startswith("Resuming incomplete job")][0]]]))
        job_infos[-1]["rule"] = rule[1][len("rule "):-1]
        job_infos[-1]["timestamp"] = datetime.strptime(rule[0][1:-1], "%a %b %d %H:%M:%S %Y")

        #TODO the order of this may be messed up if eg. "Trying to restart job" comes before "Removing output files of"
        #TODO on my local terminal, the "building dag of jobs..." until "select jobs to execute..." is yellow, the rules with their IO is green, (and other stuff I can't tell)
        # - I may be able to use these colors to split what-it-belonged-to in cases where two logs were written simultaenously to the logfile!
        for other_txt in ["Removing output files of", "failed because of an error!", "Trying to restart job"]:
            if any(other_txt in i for i in rule):
                tmpstr = rule[[n for n, i in enumerate(rule) if other_txt in i][0]:]
                leftover_text.extend([i for i in tmpstr if not "Submitted job" in i])
                rule = rule[:[n for n, i in enumerate(rule) if other_txt in i][0]]+[i for i in tmpstr if "Submitted job" in i]
        if any("Error in rule" in i for i in rule):
            tmpstr = rule[[n for n, i in enumerate(rule) if "Error in rule" in i][0] - 1:]
            leftover_text.extend([i for i in tmpstr if not "Submitted job" in i])
            rule = rule[:[n for n, i in enumerate(rule) if "Error in rule" in i][0]-1]+[i for i in tmpstr if "Submitted job" in i]


        if any(i.startswith("Will submit the following:") for i in rule):
            job_infos[-1]["submit_command"] = [i for i in rule if i.startswith("Will submit the following:")][0].split("`")[1]
            job_infos[-1]["errorfile"] = [i for i in rule if i.startswith("Error-File can be found at")][0].split("`")[1]
            job_infos[-1]["external_id"] = rule[-1].split("'")[1]
        else:
            assert any(i.startswith("Resuming incomplete job") for i in rule)
            job_infos[-1]["external_id"] = [i for i in rule if i.startswith("Resuming incomplete job")][0].split("'")[1]
            job_infos[-1]["resuming"] = True
    pure_resumes = []
    for resumed_job in [i for i in job_infos if i.get("resuming")]:
        orig_job = [i for i in job_infos if i["external_id"] == resumed_job["external_id"] and not i.get("resuming")]
        if len(orig_job) < 1:
            pure_resumes.append(resumed_job["external_id"])
        else:
            assert all(v == resumed_job[k] for k, v in orig_job.items() if k not in ["resuming", "submit_command", "errorfile", "resources", "timestamp", "resumed_at"])
            orig_job.setdefault("resumed_at", []).append(resumed_job["timestamp"])
    job_infos = [i for i in job_infos if (not i.get("resuming") or i["external_id"] in pure_resumes)]
    return todo_jobs, dones, merge_job_infos(job_infos), leftover_text


def merge_job_infos(job_infos):
    di = {}
    for job in job_infos:
        di.setdefault(job["output"], []).append(job)
    newls = []
    for vals in di.values():
        newls.append(vals[0])
        newls[-1]["resumed_at"] = {newls[-1]["external_id"]: newls[-1]["resumed_at"]} if "resumed_at" in newls[-1] else {}
        for elem in ["timestamp", "external_id", "submit_command"]:
            newls[-1][elem] = [newls[-1][elem]]
        if len(vals) >= 2:
            for i in range(1, len(vals)):
                assert all(v == vals[i][k] for k, v in vals[0].items() if k not in ["resources", "timestamp", "external_id", "submit_command", "resumed_at"])
                for elem in ["timestamp", "external_id", "submit_command"]:
                    newls[-1][elem].append(vals[i][elem])
                if "resumed_at" in vals[i]:
                    newls[-1].setdefault("resumed_at", {})[vals[i]["external_id"]] = vals[i]["resumed_at"]
    return newls

def main():
    if isfile(os.getenv("MA_SELECT_ENV_FILE", "")):
        load_dotenv(os.environ["MA_SELECT_ENV_FILE"])
    apply_dotenv_vars()
    args = parse_command_line_args()
    errorfiles, outputfiles = get_all_jobfiles(args.start_pid, args.base_dir)
    todo_jobs, dones, job_infos, leftover_text = split_all(errorfiles) #TODO consider todo_jobs and dones!
    assert len([i["jobid"] for i in job_infos]) == len(set([i["jobid"] for i in job_infos]))
    for info in job_infos:
        if info["jobid"] in dones.keys():
            info["finished_at"] = dones[info["jobid"]]
        elif not "state" in info:
            info["state"] = [get_status(i, silent=True) for i in info["external_id"]]
    # assert len([i for i in job_infos if not i.get("finished_at")]) == len(job_infos)-len(dones)
    #TODO nur weil sie nicht finished sind sind sie nicht failed...!!
    if (runnings := [i for i in job_infos if all(j == "running" for j in i.get("state"))]):
        print("The following runs are running:\n  "+"\n  ".join([f"  {r['output'].ljust(max(len(i['output']) for i in runnings))} (pid {','.join(r['external_id'])})" for r in runnings]))
        job_infos = [i for i in job_infos if i["jobid"] not in [i["jobid"] for i in runnings]]
    #TODO finished ones are not actually finished god damn it
    if (dones := [i for i in job_infos if "finished_at" in i or all(j == "success" for j in i.get("state"))]):
        print("The following ones are finished: \n   " + "\n   ".join([f"{i['output'].ljust(max(len(i['output']) for i in dones))}{('at '+str(i['finished_at'])) if 'finished_at' in i else ''}" for i in dones]))
        job_infos = [i for i in job_infos if i["jobid"] not in [i["jobid"] for i in dones]]
    #TODO there's also enqueued!
    if (fails := sorted([i for i in job_infos if not i.get("finished_at")], key=lambda x: x["output"])):
        print("The following ones failed:")
        for fail in fails:
            print(f"  {fail['output'].ljust(max(len(i['output']) for i in fails))} (pid {','.join(fail['external_id'])})")
            if args.include_errors:
                err = extract_error(read_errorfile(fail["errorfile"]))
                if err is not False: print("     " + err.replace("\n", "\n     "))
    #TODO cross-check with `check -j` command
    #TODO parse the leftover_text as well, there may be more info about fails
    #TODO use the error-files to get the respective errors of the files
    #TODO cross-check with todo_jobs OR manually run `snakemake --dry-run --ignore-all-existing-files` to make a tree (like pstree) of which branches worked an which ones didn't
    # see https://www.willmcgugan.com/blog/tech/post/rich-tree/



if __name__ == '__main__':
    main()

