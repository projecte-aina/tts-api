import multiprocessing as mp
import asyncio
import traceback
import os
import io

import numpy as np
from fastapi import APIRouter
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi import Request
from functools import partial
from itertools import chain
from pysbd import Segmenter
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from TTS.config import load_config

from server.modules.tts_request_model import TTSRequestModel
from server.audio_utils.audio_utils import generate_audio, play_audio
from server.exceptions import LanguageException, SpeakerException
from server.helper.config import ConfigONNX
from server.workers.workers import worker_onnx, worker_onnx_audio
from scripts.inference_onnx import save_wav, load_onnx_tts_unique


route = APIRouter(prefix='')
# Initialize sentence segmenter
segmenter = Segmenter(language="en")  # NEED TO BE CHANGED? THERE IS NO CATALAN BUT SPANISH
path_dir = os.path.dirname(os.path.abspath(Path(__file__).parent.parent))
route.mount("/static", StaticFiles(directory=os.path.join(path_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(path_dir, "templates"))

sessions = []


@route.get("/", response_class=HTMLResponse)
def index(request: Request):
    speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__
    return templates.TemplateResponse("index.html", {"request": request, **speaker_config_attributes})


@route.get("/startup-parameters")
def parameters():
    config = ConfigONNX()
    return JSONResponse(
        content={"speech_speed": config.speech_speed, "use_cuda": config.use_cuda},
    )


@route.get("/websocket-demo", response_class=HTMLResponse)
def websocket_demo(request: Request):
    speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__
    return templates.TemplateResponse("websocket_demo.html",{"request": request, **speaker_config_attributes})

'''
@route.get("/details", response_class=HTMLResponse)
def details(request: Request):  # ¿PARA QUÉ LO NECESITAMOS?
    config = ConfigONNX()
    model_config = load_config(config.config_path)
    if config.vocoder_config_path is not None and os.path.isfile(config.vocoder_config_path):
        vocoder_config = load_config(config.vocoder_config_path)
    else:
        vocoder_config = None

    return templates.TemplateResponse(
        "details.html",
        {"request": request,
         "show_details": config.args.show_details,
         "model_config": model_config,
         "vocoder_config": vocoder_config,
         "args": config.args.__dict__}
    )
'''


@route.get("/api/available-voices")
def available_voices():
    speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__

    return JSONResponse(
        content={"voices": list(speaker_config_attributes["speaker_ids"].keys())},
    )


def init_session_workers(model_path, session_list, idx, use_cuda):
    global sessions
    session = load_onnx_tts_unique(model_path, use_cuda=use_cuda)
    sessions[idx] = session


@route.post("/api/tts")
def tts(request: TTSRequestModel):
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

    config = ConfigONNX()

    mp.set_start_method('fork', force=True)  # spawn

    global sessions

    speaker_config_attributes = config.speakerConfigAttributes.__dict__

    speaker_id = request.voice
    text = request.text
    
    if speaker_id not in speaker_config_attributes["speaker_ids"].keys():
        raise SpeakerException(speaker_id=speaker_id)
    if request.language not in speaker_config_attributes["languages"]:
        raise LanguageException(language=request.language)

    model_path = config.model_path
    vocoder_path = config.vocoder_path
    use_cuda = config.use_cuda

    speech_rate = config.speech_speed
    temperature = config.temperature
    unique_model = config.unique_model


    wavs = worker_onnx_audio(text, speaker_id=speaker_id, model_path=model_path, unique_model=unique_model,
                                vocoder_path=vocoder_path,
                                use_aliases=speaker_config_attributes["use_aliases"],
                                new_speaker_ids=speaker_config_attributes["new_speaker_ids"], use_cuda=use_cuda,
                                temperature=temperature, speaking_rate=speech_rate)

    wavs = list(np.squeeze(wavs))
    out = io.BytesIO()
    save_wav(wavs, out)
 

    return StreamingResponse(out, media_type="audio/wav")


@route.websocket_route("/audio-stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()

    audio_queue = asyncio.Queue()

    try:
        while True:
            received_data = await websocket.receive_json()

            sentences = segmenter.segment(received_data.get("text"))
            voice = received_data.get("voice")

            # create a separate task for audio generation
            generator_task = asyncio.create_task(generate_audio(sentences, voice, audio_queue))

            # create a task for audio playing
            player_task = asyncio.create_task(play_audio(audio_queue, websocket))

            # wait for both tasks to complete
            await asyncio.gather(generator_task, player_task)

    except Exception as e:
        traceback.print_exc()
