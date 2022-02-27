#!/usr/bin/env python3
import re
import subprocess as sp
import shlex
import time
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util.sge_util import load_acctfile, read_errorfile, extract_error

logger = logging.getLogger("__name__")
logger.setLevel(40)

ORIG_ACCTFILE   = "/var/lib/gridengine/ikwgrid/common/accounting"
CUSTOM_ACCTFILE = os.getenv("MA_CUSTOM_ACCTFILE") or os.path.join(os.environ["HOME"], "custom_acctfile.yml")

stdout_print = print
print = lambda *args, **kwargs: print(*args, **kwargs, file=sys.stderr)

# WARNING this currently has no support for task array jobs
########################################################################################################################


def main():
    assert os.path.isfile(ORIG_ACCTFILE) or os.path.isfile(CUSTOM_ACCTFILE)
    job_status = get_status(sys.argv[1])
    stdout_print(job_status)


def getstatus_qstat(jobid: int, detailed: bool):
    # first try qstat to see if job is running
    # we can use `qstat -s pr -u "*"` to check for all running and pending jobs
    try:
        qstat_res = sp.check_output(shlex.split("qstat -s pr")).decode().strip()
        # skip the header using [2:]
        res = {int(x.split()[0]): x.split()[4] for x in qstat_res.splitlines()[2:]}
        # job is in an unspecified error state
        if "E" in res[jobid]:
            return "failed"
        elif res[jobid] == "qw":
            return "running" if not detailed else "enqueued"
        elif res[jobid] == "r":
            return "running"
    except sp.CalledProcessError as e:
        logger.error("qstat process error")
        logger.error(e)


def getstatus_origacctfile(jobid: int, status_attempts):
    # if the job has finished it won't appear in qstat and we should check qacct
    # this will also provide the exit status (0 on success, 128 + exit_status on fail)
    # Try getting job with scontrol instead in case sacct is misconfigured
    for i in range(status_attempts):
        try:
            qacct_res = sp.check_output(shlex.split("qacct -j {}".format(jobid)))
            exit_code = int(re.search("exit_status  ([0-9]+)", qacct_res.decode()).group(1))
            return "success" if exit_code == 0 else "failed"
        except sp.CalledProcessError as e:
            logger.warning("qacct process error")
            logger.warning(e)
            if i >= status_attempts - 1:
                return "failed"
            else:
                time.sleep(5)  # qacct can be quite slow to update on large servers


def getstatus_customacctfile(jobid: str, silent: bool):
    # `qacct` doesn't work on the IKW-grid. I asked Marc, he said "Es wird kein accounting file auf den Knoten geschrieben. Nur auf dem Master und darauf hast du keinen Zugriff"
    custom_acct = load_acctfile()
    if jobid not in custom_acct:
        return "failed"
    job_info = custom_acct[jobid]
    error_file = job_info["envvars"]["e"].format(rulename=job_info["job_properties"]["rule"], jobid=job_info["job_properties"]["jobid"])
    try:
        txt = read_errorfile(error_file)
    except FileNotFoundError:
        return "failed"

    errortype, errorstring = extract_error(txt)
    if not errortype:
        return "success"
    elif not silent:
        print("Job " + jobid + " failed because of an error: "+errorstring)
    return "failed"



def get_status(jobid: str, status_attempts=10, silent=False, detailed=False):
    for i in range(status_attempts):
        try:
            return getstatus_qstat(int(jobid), detailed)
        except (KeyError, FileNotFoundError):
            break #job doesn't appear in qstat
    if os.path.isfile(ORIG_ACCTFILE):
        return getstatus_origacctfile(int(jobid), status_attempts)
    elif os.path.isfile(CUSTOM_ACCTFILE):
        return getstatus_customacctfile(jobid, silent)
    return "failed" if not detailed else "unknown"



if __name__ == '__main__':
    main()

