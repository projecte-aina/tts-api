FROM python:3.10.12-slim
# RUN apt-get update && apt-get install -y --no-install-recommends wget gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*

# Install required packages for building eSpeak and general utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \ 
    wget \
    cmake

RUN git clone -b dev-ca https://github.com/projecte-aina/espeak-ng

RUN pip install --upgrade pip && \
 cd espeak-ng && \
 ./autogen.sh && \
 ./configure --prefix=/usr && \
 make && \
 make install


WORKDIR /app
COPY ./requirements.txt /app
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN wget -q http://share.laklak.eu/model_vits_ca/best_model.pth -P /app/models/vits_ca/
COPY . .

ARG speech_speed=1.0
ENV speech_speed $speech_speed

ARG mp_workers=2
ENV mp_workers $mp_workers

ENTRYPOINT python server/server.py --speech_speed $speech_speed --mp_workers $mp_workers
