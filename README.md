# TTS Api

RestFUL api and web interface to serve coqui TTS models

## Installation

The requirements are tested for python 3.7. In order for coqui TTS to work some dependencies should be installed.

```
sudo apt-get install libsndfile1-dev espeak-ng
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
that receives the calls from '0.0.0.0:8001', or simply
```
python server/server.py
```
which gets the calls from `0.0.0.0:8001` by default

## Docker Install and launch

To build:
```
docker build -t tts-api .
```

Also speed option can we given as a build argument, the following build makes the voices speak with `1.5` speed:

```
docker build --build-arg speed="1.5" -t tts-api-test .
```

To launch:
```
docker run --name tts -p 8001:8001 tts-api --port 8001
```
The default entrypoint puts the web interface to `http://0.0.0.0:8001/`.

#### Deployment with docker compose

```bash
make deploy
```
Example of deployment changing speech speed parameter

```bash
make deploy speed=1.6
```

The example docker-compose file shows also the build-arg usage for the speed parameter.

## Deployment via Helm

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

## Authors and acknowledgment
Developed by TeMU BSC. The code is based on Coqui TTS server.py that has a Mozilla Public License 2.0.

## License

## Project status

- [x] Conteinerized
- [ ] Improved endpoints
- [ ] Improved models
