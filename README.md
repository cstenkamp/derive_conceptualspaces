## How to get the data

* Data comes from [DESC15] and can be obtained from http://www.cs.cf.ac.uk/semanticspaces/.
* Download everything there, and make the directory-structure look like this:
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
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes
nbdime config-git --enable --global
```

### Set up Sacred

See https://sacred.readthedocs.io/en/stable/examples.html#docker-setup for the easiest way to get the MongoDB and boards to run. The */docker*-directory here is a clone of the respective *examples*-directory from the sacred-repo. To have the same `.env`-file in your local files, I can recommend PyCharm's **EnvFile**-Plugin.


