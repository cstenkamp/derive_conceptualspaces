#!/usr/bin/env python3

import os
import re
import builtins
import math
import argparse
import subprocess
import yaml
import sys
import random
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util.sge_util import load_acctfile, get_active_jobs

# use warnings.warn() rather than print() to output info in this script
# because snakemake expects the jobid to be the only output
import warnings

from snakemake import io
from snakemake.utils import read_job_properties

stdout_print = builtins.print
print = lambda *args, **kwargs: builtins.print(*args, **kwargs, file=sys.stderr)

DEFAULT_JOB_NAME = "snakemake_job"
QSUB_DEFAULTS = "-cwd -V"
CLUSTER_CONFIG = "cluster.yaml"

# SGE syntax for options is `-option [value]` and for resources is `-l name=value`
# we therefore distinguish the two in this script to make it easier to handle.
# We also define some aliases for options and resources so that the rules can
# be more expressive than a list of cryptic SGE resources.

# We additionally pickup a list of environment modules which will be loaded in the
# jobscript

OPTION_MAPPING = {
    "binding": ("binding",),
    "cwd"    : ("cwd",),
    "e"      : ("e", "error"),
    "hard"   : ("hard",),
    "j"      : ("j", "join"),
    "m"      : ("m", "mail_options"),
    "M"      : ("M", "email"),
    "notify" : ("notify",),
    "now"    : ("now",),
    "N"      : ("N", "name"),
    "o"      : ("o", "output"),
    "P"      : ("P", "project"),
    "p"      : ("p", "priority"),
    "pe"     : ("pe", "parallel_environment"),
    "pty"    : ("pty",),
    "q"      : ("q", "queue"),
    "R"      : ("R", "reservation"),
    "r"      : ("r", "rerun"),
    "soft"   : ("soft",),
    "v"      : ("v", "variable"),
    "V"      : ("V", "export_env")
}

RESOURCE_MAPPING = {
    # default queue resources
    "qname"            : ("qname",),
    "h"                : ("hostname", "h"),
    # "notify" -- conflicts with OPTION_MAPPING
    "calendar"         : ("calendar",),
    "min_cpu_interval" : ("min_cpu_interval",),
    "tmpdir"           : ("tmpdir",),
    "seq_no"           : ("seq_no",),
    "s_rt"             : ("s_rt", "soft_runtime", "soft_walltime"),
    "h_rt"             : ("h_rt", "time", "runtime", "walltime"),
    "s_cpu"            : ("s_cpu", "soft_cpu"),
    "h_cpu"            : ("h_cpu", "cpu"),
    "s_data"           : ("s_data", "soft_data"),
    "h_data"           : ("h_data", "data"),
    "s_stack"          : ("s_stack", "soft_stack"),
    "h_stack"          : ("h_stack", "stack"),           
    "s_core"           : ("s_core", "soft_core"),
    "h_core"           : ("h_core", "core"),
    "s_rss"            : ("s_rss", "soft_resident_set_size"),
    "h_rss"            : ("h_rss", "resident_set_size"),
    "pe"               : ("pe", "parallel_environment"),
    # default host resources
    "slots"            : ("slots",),
    "s_vmem"           : ("s_vmem", "soft_memory", "soft_virtual_memory"),
    "mem"              : ("h_vmem", "mem", "memory", "virtual_memory"), #it must be named "mem" for the ikw-grid
    "s_fsize"          : ("s_fsize", "soft_file_size"),
    "h_fsize"          : ("h_fsize", "file_size"),
}

TRANSLATOR_MAPPING = {
    "mem_mb": ("mem", lambda val: str(round(int(val)/1024))+"G"), #snakemake has "mem_mb" as resource, but on the grid it must be "mem"
}

def add_custom_resources(resources, resource_mapping=RESOURCE_MAPPING):
    """Adds new resources to resource_mapping.

       resources -> dict where key is sge resource name and value is a 
                    single name or a list of names to be used as aliased
    """
    for key, val in resources.items():
        if key not in resource_mapping:
            resource_mapping[key] = tuple()

        # make sure the resource name itself is an alias
        resource_mapping[key] += (key,)
        if isinstance(val, list):
            for alias in val:
                if val != key:
                    resource_mapping[key] += (alias,)
        else:
            if val != key:
                resource_mapping[key] += (val,)

def add_custom_options(options, option_mapping=OPTION_MAPPING):
    """Adds new options to option_mapping.

       options -> dict where key is sge option name and value is a single name
                  or a list of names to be used as aliased
    """
    for key, val in options.items():
        if key not in option_mapping:
            option_mapping[key] = tuple()

        # make sure the option name itself is an alias
        option_mapping[key] += (key,)
        if isinstance(val, list):
            for alias in val:
                if val != key:
                    option_mapping[key] += (alias,)
        else:
            if val != key:
                option_mapping[key] += (val,)

def parse_jobscript():
    """Minimal CLI to require/only accept single positional argument."""
    p = argparse.ArgumentParser(description="SGE snakemake submit script")
    p.add_argument("jobscript", help="Snakemake jobscript with job properties.")
    p.add_argument("-s", "--simulate", help="If you just want to simulate", default=False, action="store_true")
    return p.parse_args().jobscript, p.parse_args().simulate

def parse_qsub_defaults(parsed):
    """Unpack QSUB_DEFAULTS."""
    d = parsed.split() if type(parsed) == str else parsed
    
    options={}
    for arg in d:
        if "=" in arg:
            k,v = arg.split("=")
            options[k.strip("-")] = v.strip()
        else:
            options[arg.strip("-")] = ""
    return options

def format_job_properties(string):
    # we use 'rulename' rather than 'rule' for consistency with the --cluster-config 
    # snakemake option
    return string.format(rulename=job_properties['rule'], jobid=job_properties['jobid'])


def parse_qsub_settings(source, resource_mapping=RESOURCE_MAPPING, option_mapping=OPTION_MAPPING, translator_mapping=TRANSLATOR_MAPPING, nonrequestables=None):
    job_options = { "options" : {}, "resources" : {}}

    for skey, sval in source.items():
        if nonrequestables and skey in nonrequestables:
            continue
        found = False
        for tkey, tval in translator_mapping.items():
            if skey == tkey:
                skey = tval[0]
                sval = tval[1](sval)
        for rkey, rval in resource_mapping.items():
            if skey in rval:
                found = True
                # Snakemake resources can only be defined as integers, but SGE interprets
                # plain integers for memory as bytes. This hack means we interpret memory
                # requests as gigabytes
                if rkey in ["s_vmem", "h_vmem", "mem"] and str(sval).isnumeric():
                    job_options["resources"].update({rkey : str(sval) + 'G'})
                else:
                    job_options["resources"].update({rkey : sval})
                break
        if found: continue
        for okey, oval in option_mapping.items():
            if skey in oval:
                found = True
                job_options["options"].update({okey : sval})
                break
        if not found:
            raise KeyError(f"Unknown SGE option or resource: {skey}")

    return job_options

def load_cluster_config(path):
    """Load config to dict either from absolute path or relative to profile dir."""
    if path:
        path = os.path.join(os.path.dirname(__file__), os.path.expandvars(path))
        default_cluster_config = io.load_configfile(path)
    else:
        default_cluster_config = {}
    if "__default__" not in default_cluster_config:
        default_cluster_config["__default__"] = {}
    return default_cluster_config

def ensure_directory_exists(path):
    """Check if directory exists and create if not"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return


def update_double_dict(outer, inner):
    """Similar to dict.update() but does the update on nested dictionaries"""
    for k, v in outer.items():
        outer[k].update(inner[k])

def sge_option_string(key, val):
    if val == "":
        return f"-{key}"
    if type(val) == bool:
        return f"-{key} " + ("yes" if val else "no")
    return format_job_properties(f"-{key} {val}")

def sge_resource_string(key, val):
    if key == "pe":
        return f"-pe {val}"
    if val == "":
        return f"-l {key}"
    if type(val) == bool:
        return f"-{key}=" + ("true" if val else "false")
    if key == "mem_mb": #A snakefile automatically has the mem_mb resource (see https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#resources), but the grid wants only "mem"!
        key = "mem"
        val = str(round(int(val)/1024))+"G"
    if "mem" in key and str(val).isnumeric(): #when qsub-ing, you HAVE TO specify if it's G or not!
        val = str(val)+"G"
    if isinstance(val, list):
        random.shuffle(val) #use this for "-l h cippy01 -l h cippy02" etc (see https://doc.ikw.uni-osnabrueck.de/content/using-grid-cheat-sheet), but the ikw-grid seems to only consider the last one
        return " ".join([f"-l {key}={v}" for v in val])
    return f"-l {key}={val}"

slugify_pre = lambda txt: str(txt).replace(' ','_').replace("'","").replace('"', "")
slugify = lambda txt: ('"'+",".join(slugify_pre(i) for i in txt)+'"') if isinstance(txt, list) else slugify_pre(txt)


def check_if_alreadyscheduled(job_properties):
    to_compare = ["input", "jobid", "output", "rule", "wildcards"]
    jobs = [str(i) for i in get_active_jobs()]
    acctfile = load_acctfile()
    job_infos = {job: acctfile[job] for job in jobs if job in acctfile}
    for id, inf in job_infos.items():
        if all(inf["job_properties"].get(i) == job_properties.get(i) for i in to_compare):
            return id


def submit_job(jobscript, qsub_settings, simulate=False):
    """Submit jobscript and return jobid."""
    if not qsub_settings["resources"].get("pe", "default 1").startswith("default "):
        qsub_settings["resources"]["pe"] = "default "+qsub_settings["resources"]["pe"]
    flatten = lambda l: [item for sublist in l for item in sublist]
    batch_options = flatten([sge_option_string(k,v).split() for k, v in qsub_settings["options"].items()])
    batch_resources = flatten([sge_resource_string(k, v).split() for k, v in qsub_settings["resources"].items()])
    #I'll provide everything I know as env-vars to the script
    options_as_envvars = flatten([["-v", f"SGE_SMK_{k}={slugify(v)}"] for k, v in list(qsub_settings["options"].items())+list(qsub_settings["resources"].items()) if bool(v) or v == False])
    options_as_envvars += ["-v", f"SGE_SMK_rulename={job_properties['rule']}", "-v", f"SGE_SMK_jobid={job_properties['jobid']}"]
    if int(job_properties.get("threads", 1)) > 1:
        if not "pe" in qsub_settings["resources"]: # if, in a rule, you specify "threads" but not "resources/pe", it will set the pe from the threads.
            batch_resources.extend(["-pe", "default", str(job_properties['threads'])])
            options_as_envvars += ["-v", f"SGE_SMK_pe_threads={str(job_properties['threads']).replace(' ', '_')}"]
        #else you could check if it the threads fall between qsub_settings["resources"]["pe"][len("default "):] and warn accordingly but whatever
    if qsub_settings["resources"].get("h_rt"):
        wall_secs = sum(60**(2-i[0])*int(i[1]) for i in enumerate(qsub_settings["resources"]["h_rt"].split(":")))
        batch_options += ["-v", f"WALL_SECS={wall_secs}"]
    try:
        old_jobid = check_if_alreadyscheduled(job_properties)
        if old_jobid:
            print("This job is already scheduled under external job-id", old_jobid)
            return old_jobid
        print(f'Will submit the following: `{" ".join(["qsub", "-terse"] + batch_options + options_as_envvars + batch_resources + [jobscript])}`')
        print(f'Error-File can be found at `{os.path.join(os.getenv("MA_BASE_DIR", ""), qsub_settings["options"].get("e", "").format(rulename=job_properties["rule"], jobid=job_properties["jobid"]))}`')
        if simulate:
            jobid = None
        else:
            # -terse means only the jobid is returned rather than the normal 'Your job...' string
            jobid = subprocess.check_output(["qsub", "-terse"] + batch_options + options_as_envvars + batch_resources + [jobscript]).decode().rstrip()
    except subprocess.CalledProcessError as e:
        raise e
    except Exception as e:
        raise e
    #replacement for the accounting-file
    if "MA_CUSTOM_ACCTFILE" in os.environ:
        custom_acct = load_acctfile()
        custom_acct[jobid] = {"envvars": {i.split("=")[0][len("SGE_SMK_"):]:i.split("=")[1] for i in options_as_envvars if i != "-v"},
                              "batch_options": " ".join(batch_options), "batch_resources": " ".join(batch_resources), "job_properties": job_properties}
        with open(os.environ["MA_CUSTOM_ACCTFILE"], "w") as wfile:
            yaml.dump(custom_acct, wfile)
    return jobid

qsub_settings = { "options" : {}, "resources" : {}}

jobscript, simulate = parse_jobscript()

# get the job properties dictionary from snakemake 
job_properties = read_job_properties(jobscript)

# load the default cluster config
cluster_config = load_cluster_config(CLUSTER_CONFIG)

add_custom_resources(cluster_config["__resources__"])

add_custom_options(cluster_config["__options__"])

# qsub default arguments
update_double_dict(qsub_settings, parse_qsub_settings(parse_qsub_defaults(QSUB_DEFAULTS), nonrequestables=cluster_config["nonrequestables"]))

# cluster_config defaults
update_double_dict(qsub_settings, parse_qsub_settings(cluster_config["__default__"], nonrequestables=cluster_config["nonrequestables"]))

# resources defined in the snakemake file (note that these must be integer)
# we pass an empty dictionary for option_mapping because options should not be
# specified in the snakemake file
update_double_dict(qsub_settings, parse_qsub_settings(job_properties.get("resources", {}), option_mapping={}, nonrequestables=cluster_config["nonrequestables"]))

# get any rule specific options/resources from the default cluster config
update_double_dict(qsub_settings, parse_qsub_settings(cluster_config.get(job_properties.get("rule"), {}), nonrequestables=cluster_config["nonrequestables"]))

# get any options/resources specified through the --cluster-config command line argument
update_double_dict(qsub_settings, parse_qsub_settings(job_properties.get("cluster", {}), nonrequestables=cluster_config["nonrequestables"]))

# ensure qsub output dirs exist
for o in ("o", "e"):
    ensure_directory_exists(qsub_settings["options"][o]) if o in qsub_settings["options"] else None

# submit job and echo id back to Snakemake (must be the only stdout)
stdout_print(submit_job(jobscript, qsub_settings, simulate=simulate))

