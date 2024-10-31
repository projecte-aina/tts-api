import json
import os
from typing import Union

def update_config(config_path, velocity):
    length_scale = 1 / velocity
    with open(config_path, "r+") as json_file:
        data = json.load(json_file)
        data["model_args"]["length_scale"] = length_scale
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()

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
