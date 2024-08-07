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
        cmake \ 
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b dev-ca https://github.com/projecte-aina/espeak-ng

RUN pip install --upgrade pip && \
 cd espeak-ng && \
 ./autogen.sh && \
 ./configure --prefix=/usr && \
 make && \
 make install

RUN which espeak

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT python main.py --speech_speed ${SPEECH_SPEED} --use_cuda ${USE_CUDA}
