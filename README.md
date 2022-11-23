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
docker compose up -d --build
```
The example docker-compose file shows also the build-arg usage for the speed parameter.

## Authors and acknowledgment
Developed by TeMU BSC. The code is based on Coqui TTS server.py that has a Mozilla Public License 2.0.

## License

## Project status

- [x] Conteinerized
- [ ] Improved endpoints
- [ ] Improved models
