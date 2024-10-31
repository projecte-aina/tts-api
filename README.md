# TTS API

RestFUL api and web interface to serve matcha TTS models

## Installation

The requirements are tested for python 3.10. In order for matcha TTS to work, some dependencies should be installed.

1. Update your system's package list and install the required packages for building eSpeak and general utilities:
```bash
sudo apt update && sudo apt install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \ 
    wget \
    cmake
```

2. Clone the eSpeak-ng repository and build it:
```bash
git clone https://github.com/espeak-ng/espeak-ng
cd espeak-ng && \
 sudo ./autogen.sh && \
 sudo ./configure --prefix=/usr && \
 sudo make && \
 sudo make install
```

Later simply:

```
python -m pip install --upgrade pip
```


> [!NOTE]
> The model **best_model.onnx** is requiered, you have to download by yourself.

Download the model from HuggingFace
https://huggingface.co/projecte-aina/matxa-tts-cat-multiaccent/resolve/main/matcha_multispeaker_cat_all_opset_15_10_steps.onnx

Note: You will need a Huggingface account because the model privacity is setted to gated.

Rename the onnx model to best_model.onnx and move it to ./models/matxa_onnx folder

or download using wget

```bash
wget https://huggingface.co/projecte-aina/matxa-tts-cat-multiaccent/resolve/main/matxa_multiaccent_wavenext_e2e.onnx -O ./models/matxa_onnx/best_model.onnx
```

## Launch

tts-api uses `FastAPI` and `uvicorn` under the hood. For now, in order to launch:

```
python server/server.py --model_path models/matxa_onnx/best_model.onnx --port 8001
```
that receives the calls from `0.0.0.0:8001`, or simply
```
python server/server.py
```
which gets the calls from `0.0.0.0:8000` by default

## Usage

tts-api has three inference endpoints, two openapi ones (as can be seen via `/docs`)

* `/api/tts`: main inference endpoint
#

The example for `/api/tts` can be found in `/docs`. For the `api/tts` the call is as the following:

```
curl --location --request POST 'http://localhost:8000/api/tts' --header 'Content-Type: application/json' --data-raw '{
    "voice": "quim",
    "type": "text",
    "text": "El Consell s’ha reunit avui per darrera vegada abans de les eleccions. Divendres vinent, tant el president com els consellers ja estaran en funcions. A l’ordre del dia d’avui tampoc no hi havia l’aprovació del requisit lingüístic, és a dir la normativa que ha de regular la capacitació lingüística dels aspirants a accedir a un lloc en la Funció Pública Valenciana.",
    "language": "ca-es" }' --output tts.wav
```

## Docker launch from the hub


To launch using lastest version available on the Dockerhub:


```
docker run -p 8000:8000 projecteaina/tts-api:latest
```

[Check out the documentation available on the Dockerhub](https://hub.docker.com/r/projecteaina/tts-api)

## Docker build and launch

To build:
```
docker build -t tts-api .
```

To launch:
```
docker run -p 8000:8000 tts-api
```
The default entrypoint puts the web interface to `http://0.0.0.0:8000/`.


## Develop in docker
You can run this api with docker with reload mode that will let you watch you local changes on api.

To run in dev mode run the following command.

```bash
make dev
```



## REST API Endpoints

| **Method** | **Endpoint** | **Description**                                       |
|------------|--------------|-------------------------------------------------------|
| POST       | `/api/tts`   | Generate speech audio from text using TTS.            |

**Request Parameters:**

| **Parameter** | **Type**           | **Description**                                            |
|---------------|--------------------|------------------------------------------------------------|
| language      | string             | ISO language code (e.g., "ca-es", "ca-ba", "ca-nw", "ca-va")                          |
| voice         | string             | Name of the voice to use                                   |
| type          | string             | Type of input text ("text" or "ssml")                      |
| text          | string             | Text to be synthesized (if type is "ssml", enclose in tags) |


**NOTES:** 
- ssml format is not available yet.

**Successful Response:**

The endpoint returns a streaming response that contains the synthesized speech audio in WAV format.


**Sample Request:**

```http
POST /api/tts

{
  "voice": "speaker_id",
  "text": "Bon dia!",
  "type": "text"
}
```


#### Command line deployment arguments
| **Argument**           | **Type** | **Default**                             | **Description**                                                               |
|------------------------|----------|-----------------------------------------|-------------------------------------------------------------------------------|
| speech_speed           | float    | 1.0                                     | Change the speech speed.                                                      |



- The "speech_speed" argument refers to a parameter that adjusts the rate at which speech sounds in an audio output, with higher values resulting in faster speech, and lower values leading to slower speech.


## Deployment


### Environment Variables

To deploy this project, you will need to add the following environment variables to your .env file

`SPEECH_SPEED`

`USE_CUDA`


Example of .env file
```bash
SPEECH_SPEED=1.0
USE_CUDA=False
```


### Deployment via docker compose

#### Prerequisites

- Make

- [Docker](https://docs.docker.com/engine/install/ubuntu/)

- [Docker compose](https://docs.docker.com/compose/install/)

To deploy this app
```bash
make deploy
```

To deploy this app using GPU
```bash
make deploy-gpu
```
To stop deployment run
```bash
make stop
```
To delete deployment run
```bash
make undeploy
```

#### Deployment via Helm

The chart is still not available on any repository so you need to run this command from the repository folder.
Please, keep in mind that if you are deploying this chart to a cloud K8s instance you need to push the Docker image first
to an image registry.

Create namespace
```bash
kubectl create namespace apps
```
Deploy chart
```bash
#Run the following command from $BASE_REPO_PATH/charts/aina-tts-api path
helm upgrade --install aina-tts-api --create-namespace . 
```

You can either change the values on `values.yaml` or override them.

```bash
helm upgrade --install aina-tts-api --create-namespace \
  --set global.namespace=apps \
  --set api.image=tts-api \
  --set api.tag=latest .
```

Deploy helm chart with a different speech speed value
```bash
helm upgrade --install aina-tts-api --create-namespace \
  --set api.speech_speed=1.6 .
```

## Authors and acknowledgment
Developed by the Language Technologies Unit in Barcelona Supercomputing Center. The code is based on Coqui TTS server.py that has a Mozilla Public License 2.0.

## License
Mozilla Public License 2.0

## Project status

- [x] Conteinerized
- [x] Improved endpoints
- [x] Improved models
- [x] Speed control
- [ ] Caching

## Funding

This work is funded by the [Generalitat de
Catalunya](https://politiquesdigitals.gencat.cat/ca/inici/index.html#googtrans(ca|en))
within the framework of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).

<a target="_blank" title="Generalitat de Catalunya" href="https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina/"><img alt="Generalitat logo" src="https://bot.aina.bsc.es/logos/gene.png" width="400"></a>
