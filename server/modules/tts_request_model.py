from typing import Union
from pydantic import BaseModel, Field


class TTSRequestModel(BaseModel):
    language: Union[str, None] = "ca-es"   # ca-ba, ca-nw, ca-va
    voice: str = Field(...)
    type: str = Field(...)
    text: str = Field(..., min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "language": "ca-es",
                    "voice": "olga",
                    "type": "text",
                    "text": "hola que tal"
                }
            ]
        }
    }
