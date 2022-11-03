#!flask/bin/python
import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Union

from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from starlette.responses import JSONResponse

# global path variables
path = Path(__file__).parent / ".models.json"
path_dir = os.path.dirname(path)
manager = ModelManager(path)

# default tts model/files
models_path_rel = '../models/vits_ca'
model_ca = os.path.join(path_dir, models_path_rel, 'best_model.pth')
config_ca = os.path.join(path_dir, models_path_rel, 'config.json')

def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path",
        default=config_ca,
        type=str,
        help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_ca,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=8000, help="port to listen on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip to listen.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


# parse the args
args = create_argparser().parse_args()

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path
    speaker_ids_path = os.path.join(path_dir, models_path_rel, 'speaker_ids.json')

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)

speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)
if speaker_manager:
    new_speaker_ids = json.load(open(speaker_ids_path))
# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = FastAPI()
# in principle we don't serve static files now but we might
app.mount("/static", StaticFiles(directory=os.path.join(path_dir,"static")), name="static")
templates = Jinja2Templates(directory=os.path.join(path_dir,"templates"))


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None

class SpeakerException(Exception):
    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id

@app.exception_handler(SpeakerException)
async def speaker_exception_handler(request: Request, exc: SpeakerException):
    return JSONResponse(
        status_code=406,
        content={"message": f"{exc.speaker_id} is an unknown speaker id.", "accept": list(new_speaker_ids.keys())},
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "show_details":args.show_details,
         "use_multi_speaker":use_multi_speaker,
         #"speaker_ids":speaker_manager.ids if speaker_manager is not None else None,
         "speaker_ids":new_speaker_ids if speaker_manager is not None else None,
         "use_gst":use_gst}
    )


@app.get("/details", response_class=HTMLResponse)
async def details(request: Request):
    model_config = load_config(args.config_path)
    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        vocoder_config = None

    return templates.TemplateResponse(
        "details.html",
        {"request": request,
         "show_details": args.show_details,
         "model_config": model_config,
         "vocoder_config": vocoder_config,
         "args": args.__dict__}
    )


@app.get("/api/tts")
async def tts(speaker_id: str, text: str = Query(min_length=1)):
    if speaker_id not in new_speaker_ids.keys():
        raise SpeakerException(speaker_id=speaker_id)
    # style_wav = style_wav_uri_to_dict(style_wav)
    print(" > Model input: {}".format(text))
    print(" > Speaker Idx: {}".format(speaker_id))
    wavs = synthesizer.tts(text, speaker_name=new_speaker_ids[speaker_id])
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    print({"text": text, "speaker_idx": speaker_id})
    return StreamingResponse(out, media_type="audio/wav")

def main():
    uvicorn.run('server:app', host=args.host, port=args.port)

if __name__ == "__main__":
    main()
