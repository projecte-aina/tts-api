from starlette.responses import JSONResponse
from fastapi import Request

from server.exceptions import LanguageException, SpeakerException
from server.helper.config import ConfigONNX


async def language_exception_handler(request: Request, exc: LanguageException):
    speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__

    return JSONResponse(
        status_code=406,
        content={"message": f"{exc.language} is an unknown language id.", "accept": speaker_config_attributes["languages"]},
    )


async def speaker_exception_handler(request: Request, exc: SpeakerException):
    speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__

    return JSONResponse(
        status_code=406,
        content={"message": f"{exc.speaker_id} is an unknown speaker id.", "accept": list(speaker_config_attributes["speaker_ids"].keys())},
    )
