#!/bin/bash

############################ interpreting commandlineargs ##############################
# all other args except these are forwarded to snakemake, so you can call this with `by_config --config ...`!

killold=false
watch=false
remove=false
OPTIND=1

while getopts 'kwr' opt; do
    case $opt in
        k) killold=true ;;  # kill all other grid-jobs currently running
        w) watch=true ;;    # after submitting, watch `qstat` continually
        r) remove=true ;;   # remove all old logs and outputs
        *) echo 'Error in command line parsing' >&2
           exit 1
    esac
done
shift "$(( OPTIND - 1 ))"

############# setting the env-vars from the select_env_file ############################

if [[ -z "${MA_SELECT_ENV_FILE}" ]]; then
  echo "You need to set the env-var MA_SELECT_ENV_FILE, pointing to the main env-file!"
  exit 1;
fi
export $(cat $MA_SELECT_ENV_FILE | envsubst | xargs)
export $(cat $MA_SELECT_ENV_FILE | sed 's/{\(.*\)}/\$\1/g' | envsubst | xargs)
#replaces {ENVVAR} with $ENVVAR - that way you can use the former as syntax in the env-file to refer variables that will only be defined in the same env-file (1 nesting only)

############################### executing `-k` commandlinearg ##########################

if "$killold"; then
  $MA_CODEPATH/workflow/sge/util/qdel_all.py; sleep 3;
fi

######################### set walltime from the grid-conf-yaml #########################

source $MA_CODEPATH/workflow/sge/util/parse_yml.sh
eval $(parse_yaml $MA_GRIDCONF/cluster.yaml | grep __default___h_rt)
export WALLTIME=$(echo $__default___h_rt | tr -d '"')

eval $(parse_yaml $MA_GRIDCONF/cluster.yaml | grep runner_restarts)
export RUNNER_RESTARTS=$(echo $runner_restarts | tr -d '"')

echo "Wall-Time: $WALLTIME"
echo "Runner-restarts: $RUNNER_RESTARTS"

###################### remove logs from previous runs if demanded ######################

if "$remove"; then
  rm -rf $MA_BASE_DIR/logs
  rm -f $MA_BASE_DIR/success.file
  rm -rf $MA_BASE_DIR/.snakemake/tmp.*
  rm -f "$(pwd)"/smk_runner.*
fi


############################## finally actually run `qsub` #############################

SNAKEMAKE_ARGS="$*"
SNAKEMAKE_ARGS=${SNAKEMAKE_ARGS:-default}

echo "Arguments for Snakemake: \"$SNAKEMAKE_ARGS\""

if [[ -z "${MA_ENV_FILE}" ]]; then
    echo "ENV-FILE: $MA_ENV_FILE";
fi

qsub -V -l h_rt=$WALLTIME -v WALLTIME=$WALLTIME -v SNAKEMAKE_ARGS="$SNAKEMAKE_ARGS" -v MA_SELECT_ENV_FILE="$MA_SELECT_ENV_FILE" \
    -v MA_BASE_DIR="$MA_BASE_DIR" -v MA_CODEPATH="$MA_CODEPATH" -v MA_CONDAPATH="$MA_CONDAPATH" \
    -v MA_CUSTOM_ACCTFILE="$MA_CUSTOM_ACCTFILE" -v MA_CONFIGDIR="$MA_CONFIGDIR" -v RUNNER_RESTARTS=$RUNNER_RESTARTS \
    $MA_CODEPATH/workflow/sge/run_snakemake.sge

if "$watch"; then
  sleep 3; watch -n 5 qstat -r
fi
