## Contributing

### Set up environment

As I have some Jupyter-Notebooks in here, I am using `nbstripoutput` which ensures the outputs are not commited to Github (except for cells with the `keep_output` tag), and `nbdime` which allows for a nice `diff`-view for notebooks (also allowing for `nbdiff-web`)
```
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes
nbdime config-git --enable --global
```