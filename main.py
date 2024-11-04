import argparse
import uvicorn
import torch
import multiprocessing as mp
from pathlib import Path
import os

from server import create_app
from server.utils.argparse import MpWorkersAction


path_dir = os.path.dirname(Path(__file__))
# Set the relative paths for the default TTS model and its associated configuration
models_path_rel = f'{path_dir}/models/matxa_onnx'
model_name = 'best_model.onnx'
vocoder_name = 'best_model.onnx'
spk_ids_file = 'spk_ids.json'

model_ca = os.path.join(models_path_rel, model_name)
vocoder_ca = os.path.join(models_path_rel, vocoder_name)
ids_file_path = os.path.join(models_path_rel, spk_ids_file)


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]
    parser = argparse.ArgumentParser()
    # Args for running custom models
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_ca,
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to ONNX vocoder",
        default=vocoder_ca,
    )
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.",
                        default=ids_file_path)
    parser.add_argument("--unique_model", type=bool, help="set to True if the model is a TTS+Vocoder",
                        default=True)
    parser.add_argument("--port", type=int, default=8000, help="port to listen on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host ip to listen.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=os.environ.get("USE_CUDA", False), nargs='?', const=False,
                        help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False,
                        help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False,
                        help="Generate model detail page.")
    parser.add_argument("--speech_speed", type=float, default=os.environ.get("SPEECH_SPEED", 0.9), nargs='?', const=1.0,
                        help="Change speech speed.")
    parser.add_argument("--temperature", type=float, default=0.4, nargs='?', const=1.0,
                        help="Set temperature of inference.")
    parser.add_argument("--reload", type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="Reload on changes")
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
new_speaker_ids = None
use_aliases = None

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    speakers_file_path = args.speakers_file_path
    speaker_ids_path = os.path.join(models_path_rel, 'spk_ids.json')

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path


app = create_app(
        model_path=model_path,
        vocoder_path=vocoder_path,
        speaker_ids_path=speaker_ids_path,
        speech_speed=args.speech_speed,
        temperature=args.temperature,
        use_cuda=args.use_cuda,
        args=args
    )


def main():
    uvicorn.run('main:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    mp.set_start_method("fork")
    main()
