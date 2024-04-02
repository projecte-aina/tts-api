import multiprocessing as mp
import asyncio
import traceback
import os
import io

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
from TTS.config import load_config

from server.modules.tts_request_model import TTSRequestModel
from server.audio_utils.audio_utils import generate_audio, play_audio
from server.exceptions import LanguageException, SpeakerException
from server.helper.config import Config
from server.workers.workers import worker


route = APIRouter(prefix='')
# Initialize sentence segmenter
segmenter = Segmenter(language="en")
path_dir = os.path.dirname(os.path.abspath(Path(__file__).parent))

route.mount("/static", StaticFiles(directory=os.path.join(path_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(path_dir, "templates"))

@route.get("/", response_class=HTMLResponse)
async def index(request: Request):
    speaker_config_attributes = Config().speakerConfigAttributes.__dict__
    return templates.TemplateResponse("index.html", {"request": request, **speaker_config_attributes})


@route.get("/startup-parameters")
async def parameters():
    config = Config()
    return JSONResponse(
        content={"speech_speed": config.speech_speed, "mp_workers": config.mp_workers, "use_cuda": config.use_cuda, "use_mp": config.use_mp},
    )


@route.get("/websocket-demo", response_class=HTMLResponse)
async def websocket_demo(request: Request):
    speaker_config_attributes = Config().speakerConfigAttributes.__dict__
    return templates.TemplateResponse("websocket_demo.html",{"request": request, **speaker_config_attributes})

@route.get("/details", response_class=HTMLResponse)
async def details(request: Request):
    config = Config()
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

@route.get("/api/available-voices")
async def available_voices():
    speaker_config_attributes = Config().speakerConfigAttributes.__dict__

    return JSONResponse(
        content={"voices": list(speaker_config_attributes["speaker_ids"].keys())},
    )


@route.post("/api/tts")
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

    config = Config()

    speaker_config_attributes = config.speakerConfigAttributes.__dict__

    speaker_id = request.voice
    text = request.text
    
    if speaker_id not in speaker_config_attributes["speaker_ids"].keys():
        raise SpeakerException(speaker_id=speaker_id)
    if request.language not in speaker_config_attributes["languages"]:
        raise LanguageException(language=request.language)

    model = config.synthesizer

    if config.use_cuda or not config.use_mp:
        wavs = worker(text, speaker_id=speaker_id, model=model, use_aliases=speaker_config_attributes["use_aliases"], new_speaker_ids=speaker_config_attributes["new_speaker_ids"])
        out = io.BytesIO()
        model.save_wav(wavs, out)
    else:

        sentences = segmenter.segment(text)

        mp_workers = config.mp_workers
        worker_with_args = partial(worker, speaker_id=speaker_id, model=model, use_aliases=speaker_config_attributes["use_aliases"], new_speaker_ids=speaker_config_attributes["new_speaker_ids"])

        pool = mp.Pool(processes=mp_workers)

        results = pool.map(worker_with_args, [sentence.strip() for sentence in sentences if sentence])

        # Close the pool to indicate that no more tasks will be submitted
        pool.close()
        # Wait for all processes to complete
        pool.join()
        merged_wavs = list(chain(*results))

        out = io.BytesIO()

        model.save_wav(merged_wavs, out)

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
