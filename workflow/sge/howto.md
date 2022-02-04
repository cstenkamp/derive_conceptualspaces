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
  qconf -sq ai.q   #parameters for the `ai` group-queue (inc. walltime)
  qstat -j <jobid> #why your job isn't scheduled
  qstat -u "*"     #who else does something on the machines
  qconf -sc        #everything resource etc you can demand
  qstat -r         #full jobnames for qstat
  qstat -f -q "*"  #which queues and which machines are there for the queues
  ```
  

* TODO for grid:
  * make it easier to edit the command to be executed (snakemake default, by_config, ...), what config-file to be selected and what MA_ENV_FILE to be used
  * make functions that I know will take >90 minutes interruptible 
  * fix multiprocessing for grid
  * remove all absolute (or relative to $HOME) dirs, write instructions on what env-vars to set in the bashrc (`CODEPATH` and `DATAPATH` should be enough), make every path relative from there, and be able to run the exact same code from my machine and from the grid
  * Make good telegrambot-functionalities such that I always know what's what without having to look into logfiles
  * The grid may allocate the correct `mem`, but the log of the actual snakemake-rules on the individual nodes still say `resources: mem_mb=1000, disk_mb=1000` for rules where I didn't explicitly set it (seems like I have to explicitly create mem_mb from the mem, see https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#standard-resources)
  * Is it possible that if the smk_runner dies and restarts because of walltime, it will execute still running jobs again?! that would suck!!
    * hotfix for this would be to fix the sge-status.py file, then it doesn't die maybe?
    * die #retries werden auch ignoriert wenn der smk_runner stirbt..


For the .env-files for the grid-conf: env-vars you didn't define yet can be used in the syntax {VAR} instead of $VAR, but only 1 nested level

Sobald das envvargedöns hier richtig ist auch empfehlungen in dieses file schreiben was man in die bashrc packen kann :)


SOOO, commands:
[for sample-env-file see sample-env-file]  
`MA_ENV_FILE=siddata.env submit -kwr`
`MA_ENV_FILE=siddata.env run`
