FROM python:3.7
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ make  python3 python3-dev python3-pip python3-venv python3-wheel espeak espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists/*
RUN pip install llvmlite --ignore-installed

WORKDIR /opt
COPY ./requirements.txt /opt
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade -r /opt/requirements.txt

COPY . /opt/tts-api

CMD ["python", "tts-api/server/server.py", "--model_path", "tts-api/models/vits_ca/best_model.pth", "--config_path", "tts-api/models/vits_ca/config.json"]