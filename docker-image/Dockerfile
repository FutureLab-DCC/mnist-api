FROM python:3.10
LABEL maintainer="petronio.candido@gmail.com"
RUN apt-get update

ADD requirements.txt .

RUN pip install -r requirements.txt
RUN pip install -q flwr[simulation]
RUN apt-get git vim jupyterlab

WORKDIR /home/mnist-experiment

COPY mnist-experiment .

RUN git clone https://github.com/FutureLab-DCC/programming_api.git


