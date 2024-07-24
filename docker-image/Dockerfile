FROM python:3.10
LABEL maintainer="petronio.candido@gmail.com"
RUN apt-get update

RUN apt-get install -y git vim 

RUN pip install --upgrade pip

ADD requirements.txt .

RUN pip install -r requirements.txt
RUN pip install -q flwr[simulation]

RUN pip install --upgrade git+https://github.com/FutureLab-DCC/flautim_api.git