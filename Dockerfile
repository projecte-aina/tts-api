FROM python:3.10.12-slim

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

# download huggingface model
RUN mkdir -p /app/models/matxa_onnx
    
RUN wget https://huggingface.co/projecte-aina/matxa-tts-cat-multiaccent/resolve/main/matxa_multiaccent_wavenext_e2e.onnx -O /app/models/matxa_onnx/best_model.onnx
# install espeak-ng

RUN git clone https://github.com/espeak-ng/espeak-ng
RUN pip install --upgrade pip && \
 cd espeak-ng && \
 ./autogen.sh && \
 ./configure --prefix=/usr && \
 make && \
 make install

WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT python main.py --speech_speed ${SPEECH_SPEED} --use_cuda ${USE_CUDA}
