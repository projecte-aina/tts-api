FROM python:3.7-slim
RUN apt-get update && apt-get install -y --no-install-recommends wget gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY ./requirements.txt /opt
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade -r /opt/requirements.txt

COPY . /opt/tts-api
RUN wget -q https://huggingface.co/projecte-aina/tts-ca-coqui-vits-multispeaker/resolve/main/model/best_model.pth -P /opt/tts-api/models/vits_ca/

ARG speech_speed=1.0
ENV speech_speed $speech_speed

ENTRYPOINT python tts-api/server/server.py --speech_speed $speech_speed
