#! /usr/bin/python

import subprocess as sp
import shlex

qstat_res = sp.check_output(shlex.split("qstat -s pr")).decode().strip()

res = {int(x.split()[0]) : x.split()[4] for x in qstat_res.splitlines()[2:]}
actives = list(res.keys())

for job in actives:
  sp.check_output(shlex.split("qdel "+str(job)))

