#BUILD: `docker build -f $MA_CODE_BASE/Dockerfile --build-arg git_commit=$(git rev-parse --short HEAD) --build-arg uid=$(id -u) --build-arg gid=$(id -g) --rm --tag derive_conceptualspaces $MA_CODE_BASE`
#RUN: `docker run -it --rm --user $(id -u):$(id -g) --name derive_conceptualspaces_cont -v $MA_DATA_DIR:/opt/data --env-file $MA_ENV_FILE derive_conceptualspaces`
# with that as ma_cont eg. `MA_SNAKEMAKE_TELEGRAM=1 ma_cont snakemake --cores 3 -p  --directory /opt/data default`

ARG PYTHON_VERSION=3.10.0
FROM python:${PYTHON_VERSION}-buster

ARG uid
ARG gid
ARG git_commit

RUN apt-get update \
    && apt-get install -y bash git vim curl zsh htop tmux unzip nano

# we need nodejs >= 12 for jupyter-labextensions https://computingforgeeks.com/how-to-install-nodejs-on-ubuntu-debian-linux-mint/
RUN apt-get install -y curl dirmngr apt-transport-https lsb-release ca-certificates \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get install -y gcc g++ make \
    && apt-get install -y -f nodejs

ARG WORKDIR=/opt/derive_conceptualspaces
COPY . ${WORKDIR}
WORKDIR ${WORKDIR}

ENV PYTHONPATH=${WORKDIR}
ENV RUNNING_IN_DOCKER=1
ENV CREATE_UID=${uid}
ENV CREATE_GID=${gid}
ENV CONTAINER_GIT_COMMIT=${git_commit}

# graphviz and torch for python 3.10 are problematic, so I need to set up manual stuff
RUN pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN apt install graphviz libgraphviz-dev pkg-config python-pygraphviz -y

RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
#RUN ln -sf /usr/bin/python3 /usr/bin/python #https://stackoverflow.com/a/44967506/5122790
RUN python3 -m pip install --upgrade pip
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install -r ./requirements-dev.txt
RUN pip install -r ./requirements.txt
RUN pip install .
RUN python3 -m jupyter labextension install jupyterlab-plotly@5.3.1

RUN groupadd -g ${gid:-1000} developer \
    && useradd -l -g developer -u ${uid:-1000} -m developer
#see https://github.com/moby/moby/issues/5419#issuecomment-41478290
USER developer

RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
#https://dev.to/arctic_hen7/setting-up-zsh-in-docker-263f
RUN mkdir -p /home/developer/.antigen
RUN curl -L git.io/antigen > /home/developer/.antigen/antigen.zsh
COPY .dockershell.sh /home/developer/.zshrc
USER root
RUN chown -R developer:developer /home/developer/.antigen /home/developer/.zshrc
USER developer
RUN /bin/zsh /home/developer/.zshrc

ENV HOME=/home/developer
ENV SHELL=/bin/zsh

#build with: `docker build -f Dockerfile --build-arg uid=$(id -u) --build-arg gid=$(id -g) --build-arg git_commit=$(git rev-parse --short HEAD) --rm --tag derive_conceptualspaces .`