#docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .
#docker run -it --name derive_conceptualspaces_cont -v /home/chris/Documents/UNI_neu/Masterarbeit/data/:/opt/data derive_conceptualspaces zsh
#docker start derive_conceptualspaces_cont -i
#docker container rm derive_conceptualspaces_cont -f && docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .

ARG PYTHON_VERSION=3.9.1
FROM python:${PYTHON_VERSION}-buster

ARG uid
ARG gid
ARG git_commit

RUN apt-get update \
    && apt-get install -y bash git vim curl zsh htop tmux unzip nano

ARG WORKDIR=/opt/derive_conceptualspaces
COPY . ${WORKDIR}
WORKDIR ${WORKDIR}

ENV PYTHONPATH=${WORKDIR}
ENV RUNNING_IN_DOCKER=1
ENV CREATE_UID=${uid}
ENV CREATE_GID=${gid}
ENV CONTAINER_GIT_COMMIT=${git_commit}

RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --upgrade pip
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install -r ./requirements-dev.txt
RUN pip install -r ./requirements.txt

RUN groupadd -g ${gid:-1000} developer \
    && useradd -l -g developer -u ${uid:-1000} -m developer
#see https://github.com/moby/moby/issues/5419#issuecomment-41478290
USER developer

RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
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