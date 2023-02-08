FROM python:3.7-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends git wget gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel libsndfile1-dev \
    autoconf automake libtool pkg-config libsonic-dev ronn kramdown libpcaudio-dev && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/projecte-aina/espeak-ng --branch dev-ca --depth 1 && cd espeak-ng && \
    ./autogen.sh && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf espeak-ng

WORKDIR /opt
COPY ./requirements.txt /opt
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade -r /opt/requirements.txt

RUN wget -q http://share.laklak.eu/model_vits_ca/best_model.pth -P /opt/tts-api/models/vits_ca/

COPY . /opt/tts-api

ARG speech_speed=1.0
ENV speech_speed $speech_speed

ENTRYPOINT python tts-api/server/server.py --speech_speed $speech_speed
