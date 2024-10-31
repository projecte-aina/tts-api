from fastapi import FastAPI

from server.helper.config import ConfigONNX
from server.exceptions import LanguageException, SpeakerException
from server.exception_handler import language_exception_handler, speaker_exception_handler
from server.views.health import health
from server.views.api.api import route


def create_app(model_path, vocoder_path, speaker_ids_path, speech_speed, temperature,
               use_cuda, args) -> FastAPI:

    app = FastAPI()
    
    @app.on_event("startup")
    async def startup_event():
        config = ConfigONNX(
            model_path=model_path,
            vocoder_path=vocoder_path,
            speaker_ids_path=speaker_ids_path, 
            speech_speed=speech_speed,
            temperature=temperature,
            use_cuda=use_cuda,
            unique_model=args.unique_model
        )

    app.add_exception_handler(SpeakerException, speaker_exception_handler)
    app.add_exception_handler(LanguageException, language_exception_handler)
    app.include_router(health)
    app.include_router(route)

    return app
