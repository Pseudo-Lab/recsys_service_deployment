# pull official base image
FROM python:3.10-slim

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV AM_I_IN_A_DOCKER_CONTAINER 1
ENV MYSQLCLIENT_CFLAGS="-I/usr/include/mysql"
ENV MYSQLCLIENT_LDFLAGS="-L/usr/lib/x86_64-linux-gnu -lmysqlclient"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc libc-dev default-libmysqlclient-dev
RUN apt-get update && apt-get install -y libcurl4

COPY . /usr/src/app/

# install dependencies
RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
RUN pip install dgl==2.2.1 --find-links https://data.dgl.ai/wheels/repo.html

# 모델 다운로드
RUN python utils/download_models.py
RUN python utils/download_vectordb.py