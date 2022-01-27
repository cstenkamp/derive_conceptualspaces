## How to run this on the IKW Grid
* `qsub install_conda.sge` (ensure you don't run out of space)
* then, add an env-file, eg `siddata.env` (see `./sample.env`)
* then you can run `python -m derive_conceptualspace --env-file /home/student/c/cstenkamp/derive_conceptualspaces/config/siddata.env generate-conceptualspace show-data-info`