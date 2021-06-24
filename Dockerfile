#docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .
#docker run -it --name derive_conceptualspaces_cont -v /home/chris/Documents/UNI_neu/Masterarbeit/data/:/opt/data derive_conceptualspaces zsh
#docker start derive_conceptualspaces_cont -i
#docker container rm derive_conceptualspaces_cont -f && docker build -f Dockerfile --build-arg uid=${COMPOSE_UID:-1000} --build-arg gid=${COMPOSE_GID:-1000} --rm --tag derive_conceptualspaces .

ARG PYTHON_VERSION=3.9.1
FROM python:${PYTHON_VERSION}-buster

ARG uid
ARG gid

RUN apt-get update \
    && apt-get install -y bash git vim curl zsh htop tmux unzip nano

ARG WORKDIR=/opt/derive_conceptualspaces
COPY . ${WORKDIR}
WORKDIR ${WORKDIR}
ENV PYTHONPATH=${WORKDIR}
ENV RUNNING_IN_DOCKER=1

RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --upgrade pip
RUN ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install -r ./requirements-dev.txt
RUN pip install -r ./requirements.txt

RUN groupadd -g ${gid:-1000} developer \
    && useradd -g developer -u ${uid:-1000} -m developer
USER developer

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