#!/usr/bin/env python3

# comes from https://snakemake.readthedocs.io/en/stable/executing/cluster.html#
import os
import sys

from snakemake.utils import read_job_properties

jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)

# do something useful with the threads
threads = job_properties["threads"]

# access property defined in the cluster configuration file (Snakemake >=3.6.0)
time = job_properties["cluster"]["time"]

print(threads)
print(time)

os.system("qsub -t {threads} {script}".format(threads=threads, script=jobscript))