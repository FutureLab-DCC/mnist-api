FROM python:3.8
LABEL maintainer="petronio.candido@gmail.com"
RUN apt-get update

RUN pip install -r requirements.txt
RUN pip install -q flwr[simulation]

WORKDIR /home

RUN git clone https://github.com/FutureLab-DCC/programming_api.git

ADD ../minist-experiment/* .

