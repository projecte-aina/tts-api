import numpy as np
import os
import json

import sys
from pathlib import Path

from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager

path = Path(__file__).parent / ".models.json"
path_dir = os.path.dirname(path)
manager = ModelManager(path)

# default tts model/files
models_path_rel = '../models/vits_ca'
model_ca = os.path.join(path_dir, models_path_rel, 'best_model.pth')
config_ca = os.path.join(path_dir, models_path_rel, 'config.json')

config = VitsConfig()
config.load_json(config_ca)
vits = Vits.init_from_config(config)
vits.load_checkpoint(config,  "model.pth")

vits.export_onnx()
vits.load_onnx("coqui_vits.onnx")

text = "This is a test"
text_inputs = np.asarray(
    vits.tokenizer.text_to_ids(text, language="en"),
    dtype=np.int64,
)[None, :]

audio = vits.inference_onnx(text_inputs)
print(audio.shape)

save_wav(wav=audio[0], path="coqui_vits.wav", sample_rate=config.audio.sample_rate)