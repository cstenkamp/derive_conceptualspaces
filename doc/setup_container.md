###BUILD: 

```
docker build -f $MA_CODE_BASE/Dockerfile --build-arg git_commit=$(git rev-parse --short HEAD) --build-arg uid=$(id -u) --build-arg gid=$(id -g) --rm --tag derive_conceptualspaces $MA_CODE_BASE
```


###RUN:
```
docker run -it --rm --user $(id -u):$(id -g) --name derive_conceptualspaces_cont -v $MA_DATA_DIR:/opt/data --env-file $MA_ENV_FILE derive_conceptualspaces
```

with that aliased to `ma_cont` you can run eg. 

```
MA_SNAKEMAKE_TELEGRAM=1 ma_cont snakemake --cores 3 -p  --directory /opt/data default
```
