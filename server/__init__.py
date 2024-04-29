from fastapi import FastAPI

from server.helper.config import Config
from server.exceptions import LanguageException, SpeakerException
from server.exception_handler import language_exception_handler, speaker_exception_handler
from server.views.health import health
from server.views.api.api import route


def create_app(model_path, config_path, speakers_file_path, 
        vocoder_path, vocoder_config_path, speaker_ids_path, 
        speech_speed, mp_workers, use_cuda, use_mp, show_details, args) -> FastAPI:

    app = FastAPI()
    
    @app.on_event("startup")
    async def startup_event():
        config = Config(
            model_path=model_path, 
            config_path=config_path, 
            speakers_file_path=speakers_file_path, 
            vocoder_path=vocoder_path, 
            vocoder_config_path=vocoder_config_path, 
            speaker_ids_path=speaker_ids_path, 
            speech_speed=speech_speed, 
            mp_workers=mp_workers, 
            use_cuda=use_cuda, 
            use_mp=use_mp,
            show_details=show_details,
            args=args
        )

    app.add_exception_handler(SpeakerException, speaker_exception_handler)
    app.add_exception_handler(LanguageException, language_exception_handler)
    app.include_router(health)
    app.include_router(route)

    return app
