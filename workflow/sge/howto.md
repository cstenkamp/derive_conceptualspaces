## How to run this on the IKW Grid
* Install the Code-base: `qsub install_conda.sge` (ensure you don't run out of space)
* Add an env-file for the config, eg `siddata.env` (see `./sample.env`)
* Upload the data you need, eg. using `rsync`: `rsync -az --progress path/to/your/dataset/siddata2022/ grid:/net/projects/scratch/winter/valid_until_31_July_2022/cstenkamp/data/siddata2022 --exclude .snakemake`
* then you can run `python -m derive_conceptualspace --env-file /home/student/c/cstenkamp/derive_conceptualspaces/config/siddata.env generate-conceptualspace show-data-info` (not on the gate however! `ssh` form there onto another machine first)
* For snakemake, see https://snakemake.readthedocs.io/en/stable/executing/cluster.html 



## Set up the snakemake-profile for the grid

* see https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles and accordingly https://github.com/Snakemake-Profiles/sge : Let's install a custom snakemake-profile for the sun grid engine:
* Install cookiecutter: `pip install cookiecutter`
* Install the config from https://github.com/Snakemake-Profiles/sge: `cookiecutter https://github.com/Snakemake-Profiles/sge.git`


* **TODO: write up all the stuff I did!**

* ON THE IKW-GRID, `s_rt` **DOES NOT WORK**. I cannot check if it actually sends `SIGUSR1` because code that is called with it just doesn't produce any output.

* Useful commands:
  ```
  qhost -F         #all information about all hosts (short version: just `qhost`)
  qconf -sq all.q  #parameters for the `all` group-queue (inc. walltime)
  qstat -j <jobid> #why your job isn't scheduled
  qconf -sc        #everything resource etc you can demand
  ```