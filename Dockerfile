FROM debian:buster-slim

RUN apt-get update
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt-get install -y mecab-ipadic-utf8
RUN apt-get install -y libgl1-mesa-dev

ENV HOME="/root"

WORKDIR $HOME
RUN apt-get install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.9.16
RUN pyenv global 3.9.16

RUN python3 -m venv /opt/venv
RUN /root/.pyenv/versions/3.9.16/bin/python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt


RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY . .


# python3 -m  pipreqs.pipreqs ~/Documents/GitHub/document_recognition --force #обновление реков

# docker compose up --build

# psql -h localhost -p 7000 -U postgres -d postgres