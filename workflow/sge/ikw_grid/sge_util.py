import yaml
import os
from os.path import join
import time
import subprocess as sp
import shlex

def load_acctfile():
    if not os.path.isfile(os.environ["MA_CUSTOM_ACCTFILE"]):
        return {}
    for ntrial in range(1, 6):
        try:
            with open(os.getenv("MA_CUSTOM_ACCTFILE", join(os.getenv("HOME"), "custom_acctfile.yml")), "r") as rfile:
                custom_acct = yaml.load(rfile, Loader=yaml.SafeLoader)
            break
        except:
            time.sleep(ntrial)
    custom_acct = custom_acct if custom_acct is not None else {}
    return custom_acct


def get_active_jobs(acctfile=None):
    try:
        qstat_res = sp.check_output(shlex.split("qstat -s pr")).decode().strip()
    except FileNotFoundError:
        if acctfile is None: return []
        return list(acctfile.keys())
    res = {int(x.split()[0]) : x.split()[4] for x in qstat_res.splitlines()[2:]}
    return list(res.keys())