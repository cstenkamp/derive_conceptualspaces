rm -r /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/logs
rm /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/success.file
rm -r ~/derive_conceptualspaces/.snakemake/tmp.*
rm -r /net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/.snakemake/tmp.*
rm ~/smk_runner.*
#TODO set the env-vars (like $CODEPATH etc) here and not in the sge-file, and read the h_rt also already here such that I can give it as arg instead of as line in the .sge file
qsub -v SNAKEMAKE_ARGS="$*" ~/derive_conceptualspaces/workflow/sge/run_snakemake.sge
watch -n 5 qstat -r