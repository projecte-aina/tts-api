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
python server/server.py --model_path models/vits_ca/best_model.pth --config_path models/vits_ca/config.json
```

## Docker Install and launch

To build:
```
docker build -t tts-api .
```

To launch:
```
docker run --name tts -p 8000:8000 tts-api
```

The web application should appear in http://0.0.0.0:8000/

## Authors and acknowledgment
Developed by TeMU BSC. The code is based on Coqui TTS server.py that has a Mozilla Public License 2.0.

## License

## Project status
[x] Conteinerized
[] Improved endpoints
[] Improved models
