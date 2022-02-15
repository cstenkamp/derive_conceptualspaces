#!/usr/bin/env python3
import re
import subprocess as sp
import shlex
import sys
import time
import logging
import warnings
from os.path import dirname, join
import yaml
import os


logger = logging.getLogger("__name__")
logger.setLevel(40)

STATUS_ATTEMPTS = 20

jobid = int(sys.argv[1])
job_status = "running"

# WARNING this currently has no support for task array jobs

for i in range(STATUS_ATTEMPTS):
    # first try qstat to see if job is running
    # we can use `qstat -s pr -u "*"` to check for all running and pending jobs
    try:
        qstat_res = sp.check_output(shlex.split(f"qstat -s pr")).decode().strip()

        # skip the header using [2:]
        res = {
            int(x.split()[0]) : x.split()[4] for x in qstat_res.splitlines()[2:]
        }

        # job is in an unspecified error state
        if "E" in res[jobid]:
            job_status = "failed"
            break

        job_status = "running"
        break

    except sp.CalledProcessError as e:
        logger.error("qstat process error")
        logger.error(e)
    except KeyError as e:
        if os.path.isfile("/var/lib/gridengine/ikwgrid/common/accounting"):
            # if the job has finished it won't appear in qstat and we should check qacct
            # this will also provide the exit status (0 on success, 128 + exit_status on fail)
            # Try getting job with scontrol instead in case sacct is misconfigured
            try:
                qacct_res = sp.check_output(shlex.split(f"qacct -j {jobid}"))
                exit_code = int(re.search("exit_status  ([0-9]+)", qacct_res.decode()).group(1))
                if exit_code == 0:
                    job_status = "success"
                    break
                if exit_code != 0:
                    job_status = "failed"
                    break
            except sp.CalledProcessError as e:
                logger.warning("qacct process error")
                logger.warning(e)
                if i >= STATUS_ATTEMPTS - 1:
                    job_status = "failed"
                    break
                else:
                    # qacct can be quite slow to update on large servers
                    time.sleep(5)
        else: # `qacct` doesn't work on the IKW-grid. I asked Marc, he said "Es wird kein accounting file auf den Knoten geschrieben. Nur auf dem Master und darauf hast du keinen Zugriff"
            job_status = "success"
            if os.path.isfile(os.environ["MA_CUSTOM_ACCTFILE"]):
                with open(os.environ["MA_CUSTOM_ACCTFILE"], "r") as rfile:
                    custom_acct = yaml.load(rfile, Loader=yaml.SafeLoader)
                custom_acct = custom_acct if custom_acct is not None else {}
                if str(jobid) not in custom_acct:
                    job_info = "failed"
                    break
                job_info = custom_acct[str(jobid)]
                error_file = job_info["envvars"]["e"].format(rulename=job_info["job_properties"]["rule"], jobid=job_info["job_properties"]["jobid"])
                error_file = os.path.join(os.environ["MA_BASE_DIR"], error_file) #set in the run_snakemake.sge file ($PWD and $MA_BASE_DIR are the same)
                if not os.path.isfile(error_file):
                    job_info = "failed"
                    break
                with open(error_file, "r") as rfile:
                    txt = rfile.readlines()
                txt = [i.strip() for i in txt if i.strip()]
                startlines = [lnum for lnum, line in enumerate(txt) if line == "Building DAG of jobs..."]
                if startlines:
                    txt = txt[max(startlines):] #take only the last try into account
                if "Exiting because a job execution failed. Look above for error message" in txt:
                    print(f"Job {jobid} failed because of an error! Errorfile: {error_file}", file=sys.stderr)
                    job_status = "failed"
                    break
                elif any("Job killed after exceeding memory limits" in i for i in txt):
                    print(f"Job {jobid} failed because it reached the memory limit! Errorfile: {error_file}", file=sys.stderr)
                    job_status = "failed"
                    break

print(job_status)


# /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/logs/sge/snakejob.create_candidate_svm.8.log
# contains:
# 	```
# 	Job killed after exceeding memory limits
# 	/var/lib/gridengine/util/starter.sh: line 41:  6114 Killed                  /usr/bin/cgexec -g freezer,memory,cpuset:${CGPATH} $@
# 	```
# 	-> but for some reason, the other log-files are emtpy even though they are probably killed for the same reason!
#
# /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/.snakemake/tmp.0f0dpjxj/snakejob.create_candidate_svm.8.sh
# 	contains the line `# properties = {..... "resources": {"tmpdir": "/tmp/734385.1.training.q"}, "jobid": 8}`
# 	-> das tmp/734385 ist die job-id!
# 	-> der komplette `tmp.0f0dpjxj` ordner wird nach ende des main files gelöscht
# 	-> eigentlich sollten in dem selben direcotry auch `.jobfailed` dateien sein but they aren't, unfortunately
#
# das log vom main file enhält:
# 	* submit job 1,11,8,9 (jobid 734386ff) für die 4 param-kombis die es machen soll (wrong tmpdir though!)
# 	* versucht das accounting-file zu bekommen
# 	* says job 1 failed, says jobscript is at "/net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/.snakemake/tmp.0f0dpjxj/snakejob.create_candidate_svm.1.sh", tries to restart with new external but equal internal job id
# 	* retries creating these 4 jobs over and over, exits because execution failed, says complete log is at "/net/home/student/c/cstenkamp/derive_conceptualspaces/.snakemake/log/2022-01-28T093428.328083.snakemake.log", ends.
# 	* all error-logs are also stored in `~/derive_conceptualspaces/.snakemake/log`
#
# ==> How *could* one do an `sge-status.py` script without `qacct`:
# * check all files `/net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/.snakemake/tmp.*/snakejob.*.sh` for `"resources": {"tmpdir": "/tmp/<EXTERNALJOBID>.1.training.q"}` if the job-id is the demanded one, from that you get the jobname and the internal job id (in the filename) (`create_candidate_svm.8`)
# * With that, you can look at the file /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/logs/sge/snakejob.<JOBNAME>.<INTERNALJOBID>.log and check if it says "job killed" or the like, if yes you can return killed (and maybe even give the reason? it says `Job killed after exceeding memory limits`)
# * Also need to figure out the data-dir (my jobs may need to create an env-var where the data is stored)
#
# Another way to check if the rule ran sucessfully is to check if the output is there? I mean rn it says "Job id: 1 completed successfully, but some output files are missing"
# UPDATE: logs können auch in ~/derive_conceptualspaces/logs/sge sein!