FROM python:3.6-slim

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

# opencv dependencies
RUN  apt-get update \
  && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev



