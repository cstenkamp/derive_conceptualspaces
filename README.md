## How to run

* You need to have `git`, `docker` and `docker-compose` installed
```
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
sudo apt-get install git -y
```
* Then you can check out this repo and build the container
```
cd to/your/path
mkdir data
git clone git@github.com:cstenkamp/derive_conceptualspaces.git
cd derive_conceputalspaces
docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .
docker run -it --name derive_conceptualspaces_cont -v $(realpath ../data):/opt/data derive_conceptualspaces bash
```
* ...which brings you into the shell of the container, in which you can then run
```

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

### Set up Sacred

See https://sacred.readthedocs.io/en/stable/examples.html#docker-setup for the easiest way to get the MongoDB and boards to run. The */docker*-directory here is a clone of the respective *examples*-directory from the sacred-repo. To have the same `.env`-file in your local files, I can recommend PyCharm's **EnvFile**-Plugin.


