#! /home/student/c/cstenkamp/miniconda/envs/derive_conceptualspaces/bin/python3

import yaml
import os
from os.path import join
import time
import subprocess as sp
import shlex

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




if __name__ == '__main__':
    print(get_enqueued_detailed())