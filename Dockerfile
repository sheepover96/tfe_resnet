FROM nvidia/cuda:9.0-base

RUN apt-get install -y --no-install-recommends \
    sudo ssh \
    build-essential \
    git curl wget vim tree \
    python3-dev python3-pip python3-setuptools

ARG user_name=okura
