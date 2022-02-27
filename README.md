# Derive Conceptual spaces

Master's thesis deriving conceptual spaces from course data, following [DESC15] and others.

## How to run

First install `git`, `docker` and `docker-compose` - see file `docs/install_docker.md`

### Windows
After installing, use the Explorer to change to a directory of your choice. Download the `install_windows.bat` file from [here](https://raw.githubusercontent.com/cstenkamp/derive_conceptualspaces/main/install_windows.bat) (right click -> save as), paste it into that directory and execute it by double-clicking. Eventually it will tell you to read the instructions and subsequently opens a text-editor in which you have to enter some data. Make sure to do that and to close the editor afterwards, and press <kbd>Enter</kbd> to continue.

### Linux/Mac
* Check out this repo, set the env-file with the variables (see TODO xyz), and build the container (see below)
* To use the Google Translate API you need a gcloud-credentials-json.  
* To download data, you need an account for the Myshare of the University of Osnabrück as the data is currently hosted there. The credentials for the account also go into the env-file  

[TODO update these instructions!!!]
```
cd to/your/path
mkdir data
#put your gcloud-credentials file under the name `gcloud_tools_key.json` into the data-directory
git clone https://github.com/cstenkamp/derive_conceptualspaces.git
cd derive_conceptualspaces
cp docker/sample.env docker/.env
#edit the .env to enter correct passwords etc
docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .
docker run -it --name derive_conceptualspaces_cont -v $(realpath ../data):/opt/data --env-file ./docker/.env  derive_conceptualspaces zsh
```
* ...which brings you into the shell of the container, in which you can then start downloading data and run everything.

### All OS
[TODO update these instructions!!!]
```
python scripts/util/download_model.py
python scripts/create_siddata_dataset.py translate_descriptions
```


## How to get the non-course-data

* Data comes from [DESC15] and can be obtained from http://www.cs.cf.ac.uk/semanticspaces/.
* Download everything there, and make the directory-structure look like this (TODO - write script that uses `wget` and moves correctly!)
```
    movies
        classesGenres
        classesKeywords
        classesRatings
        d20
            DirectionsHeal
            clusters20.txt
            films20.mds
            films20.projected
            projections20.data
        d50
            ...
        d100
            ...
        d200
            ...
        Tokens
        filmNames.txt
        tokens.json
    wines
        classes
        d20
            ...
        d50
            ...
        d100
            ...
        d200
            ...
        Tokens
        wineNames.txt
    places
        ...
```
## Contributing

### Set up development-environment

As I have some Jupyter-Notebooks in here, I am using `nbstripoutput` which ensures the outputs are not commited to Github (except for cells with the `keep_output` tag), and `nbdime` which allows for a nice `diff`-view for notebooks (also allowing for `nbdiff-web`)
```
pip install -r requirements-dev.txt
nbstripout --install --attributes .gitattributes
nbdime config-git --enable --global
```

### How to run notebooks

To render plotly correctly in notebooks, install the jupyterlab extension, then run the notebook (like this exemplary for a conda-env):
```
conda run -n create_cs python -m jupyter labextension install jupyterlab-plotly@5.3.1
conda run -n create_cs python -m jupyter lab --notebook-dir=/path/to/project-root
```


### Set up Sacred

See https://sacred.readthedocs.io/en/stable/examples.html#docker-setup for the easiest way to get the MongoDB and boards to run. The */docker*-directory here is a clone of the respective *examples*-directory from the sacred-repo. To have the same `.env`-file in your local files, I can recommend PyCharm's **EnvFile**-Plugin.

## References

[DESC15] J. Derrac and S. Schockaert, Inducing semantic relations from conceptual spaces: a data-driven approach to plausible reasoning. 2015.