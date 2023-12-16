# TTS API

RestFUL api and web interface to serve coqui TTS models

## Installation

The requirements are tested for python 3.10. In order for coqui TTS to work, some dependencies should be installed.

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
git clone -b dev-ca https://github.com/projecte-aina/espeak-ng
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

In order to synthesize, the actual model needs to be downloaded and the paths in the config file need to be changed (replacing `/opt` with the top directory of the repository). The model can be downloaded from [http://share.laklak.eu/model_vits_ca/best_model.pth](http://share.laklak.eu/model_vits_ca/best_model.pth) to the models directory.

## Launch

tts-api uses `FastAPI` and `uvicorn` under the hood. For now, in order to launch:

```
python server/server.py --model_path models/vits_ca/best_model.pth --config_path models/vits_ca/config.json --port 8001
```
that receives the calls from `0.0.0.0:8001`, or simply
```
python server/server.py
```
which gets the calls from `0.0.0.0:8000` by default

## Usage

tts-api has three inference endpoints, two openapi ones (as can be seen via `/docs`) and one websocket endpoint:

* `/api/tts`: main inference endpoint
* `/audio-stream`: websocket endpoint; capable of doing async inference, as soon as the first segment is synthesized the audio starts streaming.

The example for `/api/tts` can be found in `/docs`. The websocket request is contingent on the communication with the client, hence we provide an example client at the `/websocket-demo` endpoint. For the `api/tts` the call is as the following:

```
curl --location --request POST 'http://localhost:8080/api/tts' --header 'Content-Type: application/json' --data-raw '{
    "voice": "f_cen_81",
    "type": "text",
    "text": "El Consell s’ha reunit avui per darrera vegada abans de les eleccions. Divendres vinent, tant el president com els consellers ja estaran en funcions. A l’ordre del dia d’avui tampoc no hi havia l’aprovació del requisit lingüístic, és a dir la normativa que ha de regular la capacitació lingüística dels aspirants a accedir a un lloc en la Funció Pública Valenciana.",
    "language": "ca-es" }' --output tts.wav
```

## Docker launch from the hub


To launch using lastest version available on the Dockerhub:


```
docker run --shm-size=1gb -p 8080:8000 projecteaina/tts-api:latest
```

[Check out the documentation available on the Dockerhub](https://hub.docker.com/r/projecteaina/tts-api)

## Docker build and launch

To build:
```
docker build -t tts-api .
```

To launch:
```
docker run --shm-size=1gb -p 8080:8000 tts-api
```
The default entrypoint puts the web interface to `http://0.0.0.0:8080/`.



## REST API Endpoints

| **Method** | **Endpoint** | **Description**                                       |
|------------|--------------|-------------------------------------------------------|
| POST       | `/api/tts`   | Generate speech audio from text using TTS.            |

**Request Parameters:**

| **Parameter** | **Type**           | **Description**                                            |
|---------------|--------------------|------------------------------------------------------------|
| language      | string             | ISO language code (e.g., "ca-es")                          |
| voice         | string             | Name of the voice to use                                   |
| type          | string             | Type of input text ("text" or "ssml")                      |
| text          | string             | Text to be synthesized (if type is "ssml", enclose in tags) |


**NOTES:** 
- ssml format is not available yet.
- Currently, only "ca-es" language is supported, and will be applied by default

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
| mp_workers             | int      | 2                                       | Number of CPUs used for multiprocessing.                                      |
| speech_speed           | float    | 1.0                                     | Change the speech speed.                                                      |

- mp_workers: the "mp_workers" argument specifies the number of separate processes used for inference. For example, if mp_workers is set to 2 and the input consists of 2 sentences, there will be a process assigned to each sentence, speeding up  inference.

- The "speech_speed" argument refers to a parameter that adjusts the rate at which speech sounds in an audio output, with higher values resulting in faster speech, and lower values leading to slower speech.


## Deployment


### Environment Variables

To deploy this project, you will need to add the following environment variables to your .env file

`SPEECH_SPEED`

`MP_WORKERS`

`USE_CUDA`

`USE_MP`

`SHM_SIZE`


Example of .env file
```bash
SPEECH_SPEED=1.0
MP_WORKERS=4
USE_CUDA=False
USE_MP=True
SHM_SIZE=2gb
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
