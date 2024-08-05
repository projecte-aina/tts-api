from pathlib import Path
from TTS.utils.manage import ModelManager
# from lingua_franca import load_language # Lingua franca

import argparse
import uvicorn
import torch
import multiprocessing as mp
import sys
import os

from server import create_app
from server.utils.argparse import MpWorkersAction
from server.utils.utils import update_config

# Set global paths
# Determine the current script's directory and set up paths related to the model
path = Path(__file__).parent / "server" /".models.json"
path_dir = os.path.dirname(path)

# Initialize the model manager with the aforementioned path
manager = ModelManager(path)

# Set the relative paths for the default TTS model and its associated configuration
models_path_rel = '../models/vits_ca'
model_ca = os.path.join(path_dir, models_path_rel, 'best_model.pth')
config_ca = os.path.join(path_dir, models_path_rel, 'config.json')

# Load lingua franca language
# load_language('ca-es')


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    # Create an argument parser to handle command-line arguments
    # The parser setup seems incomplete and might be continued in the next section of the code.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released "
                                                                       "vocoder models.")
    # Args for running custom models
    parser.add_argument(
        "--config_path",
        default=config_ca,
        type=str,
        help="Path to model config file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_ca,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you "
             "installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=8000, help="port to listen on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip to listen.")
    parser.add_argument("--use_mp", type=convert_boolean, default=False, nargs='?', const=True, help="true to use Python multiprocessing.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, nargs='?', const=False, help="true to use CUDA.")
    parser.add_argument("--mp_workers", action=MpWorkersAction, type=int, default=mp.cpu_count(), nargs='?', const=mp.cpu_count(), help="number of CPUs used for multiprocessing")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    parser.add_argument("--speech_speed", type=float, default=1.0, nargs='?', const=1.0, help="Change speech speed.")
    parser.add_argument("--reload", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Reload on changes")
    return parser


# parse the args
args = create_argparser().parse_args()
print("args =========", args)
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


app = create_app(
        model_path = model_path, 
        config_path = config_path, 
        speakers_file_path = speakers_file_path, 
        vocoder_path = vocoder_path, 
        vocoder_config_path = vocoder_config_path, 
        speaker_ids_path = speaker_ids_path, 
        speech_speed = args.speech_speed, 
        mp_workers = args.mp_workers, 
        use_cuda = args.use_cuda, 
        use_mp = args.use_mp,
        show_details=args.show_details,
        args=args
    )


def main():
    uvicorn.run('main:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    mp.set_start_method("fork")
    main()
