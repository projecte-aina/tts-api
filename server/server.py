import argparse
import asyncio
import io
import json
import os
import sys
import traceback
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Union

import torch
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket

from utils.argparse import MpWorkersAction

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
    parser.add_argument("--mp_workers", action=MpWorkersAction, type=int, default=2,
                        help="number of CPUs used for multiprocessing")
    # parser.add_argument("--mp_workers", action=MpWorkersAction ,type=int, default=mp.cpu_count(), help="number of CPUs used for multiprocessing")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    parser.add_argument("--speech_speed", type=float, default=1.0, help="Change speech speed.")
    return parser


def update_config(config_path, velocity):
    length_scale = 1 / velocity
    with open(config_path, "r+") as json_file:
        data = json.load(json_file)
        data["model_args"]["length_scale"] = length_scale
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()


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
# new_speaker_ids = None
# use_aliases = None

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

# CASE4: change speaker speed
if args.speech_speed != 1.0:
    update_config(config_path, args.speech_speed)

app = FastAPI()
# in principle we don't serve static files now but we might
app.mount("/static", StaticFiles(directory=os.path.join(path_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(path_dir, "templates"))


@app.on_event("startup")
async def startup_event():
    app.state.synthesizer = Synthesizer(
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

    app.state.SpeakerConfigAttributes = SpeakerConfigAttributes()


class SpeakerConfigAttributes:
    def __init__(self):
        self.use_multi_speaker = None
        self.speaker_ids = None
        self.speaker_manager = None
        self.languages = None
        self.new_speaker_ids = None
        self.use_aliases = True
        self.use_gst = None

        self.setup_speaker_attributes()

    def setup_speaker_attributes(self):
        # global new_speaker_ids, use_aliases

        model = app.state.synthesizer

        use_multi_speaker = hasattr(model.tts_model, "num_speakers") and (
                model.tts_model.num_speakers > 1 or model.tts_speakers_file is not None
        )

        speaker_manager = getattr(model.tts_model, "speaker_manager", None)
        if speaker_manager:
            self.new_speaker_ids = json.load(open(speaker_ids_path))

        if self.use_aliases:
            self.speaker_ids = self.new_speaker_ids
        else:
            self.speaker_ids = speaker_manager.ids

        self.languages = ['ca-es']

        # TODO: set this from SpeakerManager
        self.use_gst = model.tts_config.get("use_gst", False)

        self.use_multi_speaker = use_multi_speaker
        self.speaker_manager = speaker_manager

    def get_use_multi_speaker(self):
        return self.use_multi_speaker

    def get_speaker_ids(self):
        return self.speaker_ids

    def get_speaker_manager(self):
        return self.speaker_manager

    def get_languages(self):
        return self.languages

    def get_new_speaker_ids(self):
        return self.new_speaker_ids

    def get_use_aliases(self):
        return self.use_aliases

    def get_use_gst(self):
        return self.use_gst


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
    speaker_config_attrib = app.state.SpeakerConfigAttributes
    speaker_ids = speaker_config_attrib.get_speaker_ids()

    return JSONResponse(
        status_code=406,
        content={"message": f"{exc.speaker_id} is an unknown speaker id.", "accept": list(speaker_ids.keys())},
    )


class LanguageException(Exception):
    def __init__(self, language: str):
        self.language = language


@app.exception_handler(LanguageException)
async def speaker_exception_handler(request: Request, exc: LanguageException):
    speaker_config_attrib = app.state.SpeakerConfigAttributes
    languages = speaker_config_attrib.get_languages()

    return JSONResponse(
        status_code=406,
        content={"message": f"{exc.language} is an unknown language id.", "accept": languages},
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    speaker_config_attrib = app.state.SpeakerConfigAttributes
    use_multi_speaker = speaker_config_attrib.get_use_multi_speaker()
    speaker_ids = speaker_config_attrib.get_speaker_ids()
    speaker_manager = speaker_config_attrib.get_speaker_manager()
    use_gst = speaker_config_attrib.get_use_gst()

    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "show_details": args.show_details,
         "use_multi_speaker": use_multi_speaker,
         # "speaker_ids":speaker_manager.ids if speaker_manager is not None else None,
         "speaker_ids": speaker_ids if speaker_manager is not None else None,
         "use_gst": use_gst}
    )


@app.get("/websocket-demo", response_class=HTMLResponse)
async def index(request: Request):
    speaker_config_attrib = app.state.SpeakerConfigAttributes
    use_multi_speaker = speaker_config_attrib.get_use_multi_speaker()
    speaker_ids = speaker_config_attrib.get_speaker_ids()
    speaker_manager = speaker_config_attrib.get_speaker_manager()
    use_gst = speaker_config_attrib.get_use_gst()

    return templates.TemplateResponse(
        "websocket_demo.html",
        {"request": request,
         "show_details": args.show_details,
         "use_multi_speaker": use_multi_speaker,
         # "speaker_ids":speaker_manager.ids if speaker_manager is not None else None,
         "speaker_ids": speaker_ids if speaker_manager is not None else None,
         "use_gst": use_gst}
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


def worker(sentence, speaker_id, model, use_aliases, new_speaker_ids):

    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))

    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
    else:
        input_speaker_id = speaker_id

    wavs = model.tts(sentence, input_speaker_id)

    return wavs


class TTSRequestModel(BaseModel):
    language: Union[str, None] = "ca-es"
    voice: str
    type: str
    text: str = Field(..., min_length=1)


@app.post("/api/tts")
async def tts(request: TTSRequestModel):
    """
       Text-to-Speech API endpoint.

       This endpoint receives a TTSRequestModel object containing the voice and text to be synthesized. It performs the
       necessary processing to generate the corresponding speech audio and streams it back as a WAV audio file.

       Parameters:
       - request: TTSRequestModel - An object containing the voice and text data for synthesis.

       Returns:
       - StreamingResponse: A streaming response object that contains the synthesized speech audio as a WAV file.

       Raises:
       - SpeakerException: If the specified speaker ID is invalid.
       - LanguageException: If the specified language is not supported.

       """

    speaker_config_attrib = app.state.SpeakerConfigAttributes
    speaker_ids = speaker_config_attrib.get_speaker_ids()
    languages = speaker_config_attrib.get_languages()
    use_aliases = speaker_config_attrib.get_use_aliases()
    new_speaker_ids = speaker_config_attrib.get_new_speaker_ids()

    speaker_id = request.voice
    text = request.text

    if speaker_id not in speaker_ids.keys():
        raise SpeakerException(speaker_id=speaker_id)
    if request.language not in languages:
        raise LanguageException(language=request.language)

    model = app.state.synthesizer

    sentences = text.split('.')

    mp_workers = args.mp_workers
    worker_with_args = partial(worker, speaker_id=speaker_id, model=model, use_aliases=use_aliases, new_speaker_ids=new_speaker_ids)

    pool = mp.Pool(processes=mp_workers)

    results = pool.map(worker_with_args, [sentence.strip() + '.' for sentence in sentences if sentence])
    # Close the pool to indicate that no more tasks will be submitted
    pool.close()
    # Wait for all processes to complete
    pool.join()
    merged_wavs = list(chain(*results))

    out = io.BytesIO()

    model.save_wav(merged_wavs, out)

    return StreamingResponse(out, media_type="audio/wav")


async def play_audio(queue: asyncio.Queue, websocket: WebSocket):
    while True:
        # get the next audio chunk from the queue
        audio_chunk = await queue.get()

        # check if this is the end of the stream
        if audio_chunk is None:
            break

        # send the audio chunk to the client
        await websocket.send_bytes(audio_chunk)
        # print a message for debugging
        # print(f"Sent audio chunk of {len(audio_chunk)} bytes")
        # receive any data from the client (this will return None if the connection is closed)
        # TODO needs a timeout here in case the audio is not played (or finished?) within a given time
        data = await websocket.receive()
        # check if the connection is closed
        if data is None:
            break


def generate(sentence, speaker_ids, model, new_speaker_ids, use_aliases, speaker_id="f_cen_81"):
    print(f"Processing sentence: {sentence}")

    if speaker_id not in speaker_ids.keys():
        raise SpeakerException(speaker_id=speaker_id)
    # style_wav = style_wav_uri_to_dict(style_wav)
    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))
    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
    else:
        input_speaker_id = speaker_id

    wavs = model.tts(sentence, speaker_name=input_speaker_id)

    out = io.BytesIO()

    print(f"Out: {out}")
    model.save_wav(wavs, out)

    return out

@app.websocket_route("/audio-stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    audio_queue = asyncio.Queue()

    try:
        while True:
            received_data = await websocket.receive_json()

            sentences = received_data.get("text").split('.')
            speaker_id = received_data.get("speaker_id")

            # create a separate task for audio generation
            generator_task = asyncio.create_task(generate_audio(sentences, speaker_id, audio_queue))

            # create a task for audio playing
            player_task = asyncio.create_task(play_audio(audio_queue, websocket))

            # wait for both tasks to complete
            await asyncio.gather(generator_task, player_task)

    except Exception as e:
        traceback.print_exc()


async def generate_audio(sentences, speaker_id, audio_queue):
    model = app.state.synthesizer

    speaker_config_attrib = app.state.SpeakerConfigAttributes
    speaker_ids = speaker_config_attrib.get_speaker_ids()
    new_speaker_ids = speaker_config_attrib.get_new_speaker_ids()
    use_aliases = speaker_config_attrib.get_use_aliases()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        for sentence in sentences:
            if sentence:
                content = await loop.run_in_executor(executor, generate, sentence, speaker_ids, model, new_speaker_ids,
                                                     use_aliases, speaker_id)
                await audio_queue.put(content)

    await audio_queue.put(None)  # signal that we're done generating audio


def main():
    uvicorn.run('server:app', host=args.host, port=args.port)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    mp.set_start_method("fork")
    main()
