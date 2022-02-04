#!/bin/bash

############# setting the env-vars from the select_env_file ############################

if [[ -z "${MA_SELECT_ENV_FILE}" ]]; then
  echo "You need to set the env-var MA_SELECT_ENV_FILE, pointing to the main env-file!"
  exit 1;
fi
export $(cat $MA_SELECT_ENV_FILE | envsubst | xargs)
export $(cat $MA_SELECT_ENV_FILE | sed 's/{\(.*\)}/\$\1/g' | envsubst | xargs)
#replaces {ENVVAR} with $ENVVAR - that way you can use the former as syntax in the env-file to refer variables that will only be defined in the same env-file (1 nesting only)

################################## set arguments #######################################

if [[ -f "$MA_ENV_FILE" ]]; then
    echo "ENV-FILE: $MA_ENV_FILE";
else
   export MA_ENV_FILE=$MA_CONFIGDIR/$MA_ENV_FILE;
   echo "ENV-FILE: $MA_ENV_FILE"
fi

SNAKEMAKE_ARGS="$*"
SNAKEMAKE_ARGS=${SNAKEMAKE_ARGS:-default}

################################## run snakemake #######################################

source $MA_CONDAPATH/bin/activate derive_conceptualspaces
export PYTHONPATH=$MA_CONDAPATH:$PYTHONPATH
cd $MA_CODEPATH

snakemake --directory $MA_BASE_DIR --cores 1 -p "$SNAKEMAKE_ARGS" --unlock
snakemake --directory $MA_BASE_DIR --cores 1 -p "$SNAKEMAKE_ARGS"
