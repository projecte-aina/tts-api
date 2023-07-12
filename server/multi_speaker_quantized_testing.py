import numpy as np
import os
import json
import sys
import time
import torch
import onnx

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import onnx
import onnxruntime as  ort
from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange

import onnxruntime
import neural_compressor

from onnxruntime.quantization import quantize_dynamic, QuantType
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.experimental import ModelConversion, common
from neural_compressor import quantization

from pathlib import Path
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager


import argparse
import asyncio
import random
import io
import json
import os
import sys
import traceback
import time
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

torch.set_num_threads(1)

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

torch.set_num_threads(1)

config = VitsConfig()
config.load_json("/home/mllopart/PycharmProjects/ttsAPI/tts-api/models/vits_ca/config.json")
vits = Vits.init_from_config(config)
vits.load_checkpoint(config,  "/home/mllopart/PycharmProjects/ttsAPI/tts-api/models/vits_ca/best_model.pth")

vits.export_onnx()
vits.load_onnx("coqui_vits.onnx")

model = "/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/coqui_vits.onnx"

configuration = PostTrainingQuantConfig(approach="dynamic")
q_model = quantization.fit(model, configuration)
q_model.save("/home/mllopart/PycharmProjects/ttsAPI/tts-api/server/q_model.onnx")
vits.load_onnx("q_model.onnx")


new_speaker_ids = json.load(open(speaker_ids_path))

input_speaker_id = new_speaker_ids["f_cen_81"]

num_speakers = range(265)

text1 = "L'exploració de l'univers és una vasta i profunda odissea que l'ésser humà ha intentat comprendre i conquerir des de temps immemorials. Les estrelles, els planetes, les galàxies i els enigmes que amaguen han alimentat la nostra curiositat i nosaltres, com a espècie, hem arribat a una etapa on tenim les eines i la tecnologia per aprofundir en aquesta indagació amb més detall que mai. Des del llançament del primer satèl·lit artificial, Sputnik, l'any 1957, l'espècie humana ha fet grans avenços en l'exploració de l'espai, des dels viatges tripulats a la Lluna fins a les missions no tripulades a Mart i més enllà."

text_inputs1 = np.asarray(
    vits.tokenizer.text_to_ids(text1, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio1 = vits.inference_onnx(text_inputs1, speaker_id=random.choice(num_speakers))
end = time.time()
print("Inference 1 Time Taken: ", end - start, " seconds")

print(audio1.shape)
save_wav(wav=audio1[0], path="ONNX_1.wav", sample_rate=config.audio.sample_rate)

text2 = "Els telescopis, satèl·lits, sondes i robots que enviem a l'espai estan ampliant la nostra comprensió del cosmos com mai abans. Observem l'univers a diferents longituds d'ona, des de la llum visible fins a l'infraroig, les ones de ràdio, els raigs X i els raigs gamma, cada una oferint una visió única del cosmos. Les dades obtingudes ens han permès descobrir planetes més enllà del nostre sistema solar, comprendre la naturalesa dels forats negres i estudiar la matèria fosca i l'energia fosca, dues entitats misterioses que semblen dominar la major part de l'univers."

text_inputs2 = np.asarray(
    vits.tokenizer.text_to_ids(text2, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio2 = vits.inference_onnx(text_inputs2, speaker_id=random.choice(num_speakers))
end = time.time()
print("Inference 2 Time Taken: ", end - start, " seconds")

print(audio2.shape)
save_wav(wav=audio2[0], path="ONNX_2.wav", sample_rate=config.audio.sample_rate)

text3 = "Malgrat aquests increïbles avenços, l'univers continua sent un lloc de grans misteris i incomprensions. Continuem buscant vida més enllà de la Terra, una tasca que ens ha fet posar els ulls en planetes com Mart i les llunes d'enormes planetes gasosos com Júpiter i Saturn. A més, l'univers continua expandint-se a una velocitat cada cop més ràpida, un fenomen que desafia la nostra comprensió de la física i de la naturalesa de l'espai i el temps. La recerca de respostes a aquests enigmes és el que ens motiva a continuar explorant, amb la mirada fixa a l'espai, la nostra darrera frontera."

text_inputs3 = np.asarray(
    vits.tokenizer.text_to_ids(text3, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio3 = vits.inference_onnx(text_inputs3, speaker_id=random.choice(num_speakers))
end = time.time()
print("Inference 3 Time Taken: ", end - start, " seconds")

print(audio3.shape)
save_wav(wav=audio3[0], path="ONNX_3.wav", sample_rate=config.audio.sample_rate)
