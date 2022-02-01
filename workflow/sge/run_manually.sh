export DATAPATH="/net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data"
export CODEPATH="$HOME/derive_conceptualspaces"
export CONDAPATH="$HOME/miniconda/bin"
export CUSTOM_ACCTFILE=$HOME/custom_acctfile.yml
export MA_SELECT_ENV_FILE="$CODEPATH/config/_select_env_grid.env"

source $CONDAPATH/activate derive_conceptualspaces

cd $CODEPATH

(export $(cat $MA_SELECT_ENV_FILE | xargs) && PYTHONPATH=$(realpath .):$PYTHONPATH MA_LANGUAGE=en snakemake --directory $DATAPATH --cores 1 -p by_config --configfile ./config/derrac2015_edited.yml --unlock)
(export $(cat $MA_SELECT_ENV_FILE | xargs) && PYTHONPATH=$(realpath .):$PYTHONPATH MA_LANGUAGE=en snakemake --directory $DATAPATH --cores 1 -p by_config --configfile ./config/derrac2015_edited.yml --keep-going)
