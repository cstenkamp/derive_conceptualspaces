#!/bin/bash

############################# interpreting `-k` commandlinearg #########################

killold=false
OPTIND=1

while getopts 'k' opt; do
    case $opt in
        k) killold=true ;;
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
export WALLTIME=$__default___h_rt
export WALL_SECS=$(echo $WALLTIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }' )

echo "Wall-Time: $WALLTIME"

#########################

/bin/bash $MA_CODEPATH/workflow/sge/run_snakemake.sge
exit 0;

rm -r /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/logs
rm /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/success.file
rm -r ~/derive_conceptualspaces/.snakemake/tmp.*
rm -r /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/.snakemake/tmp.*
rm ~/smk_runner.*
#TODO set the env-vars (like $CODEPATH etc) here and not in the sge-file, and read the h_rt also already here such that I can give it as arg instead of as line in the .sge file
qsub -v SNAKEMAKE_ARGS="$*" MA_SELECT_ENV_FILE="$MA_SELECT_ENV_FILE" $MA_CODEPATH/workflow/sge/run_snakemake.sge
watch -n 5 qstat -r