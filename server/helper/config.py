# from argparse import Namespace
import json
# from TTS.utils.synthesizer import Synthesizer

from server.helper.singleton import Singleton
from scripts.inference_onnx import load_onnx_tts

class ConfigONNX(metaclass=Singleton):
    def __init__(self, model_path, vocoder_path, speaker_ids_path,
                 speech_speed, temperature, use_cuda, unique_model) -> None:
        self.speech_speed = speech_speed
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.vocoder_path = vocoder_path
        self.unique_model = unique_model

        # self.model_tts, self.vocoder = load_onnx_tts(model_path=model_path, vocoder_path=vocoder_path, use_cuda=False)

        # speakers_id_path es el JSON con los nombres de los speakers
        self.speakerConfigAttributes = SpeakerConfigAttributes(speaker_ids_path)


class SpeakerConfigAttributes:
    def __init__(self, speaker_ids_path) -> None:
        self.use_multi_speaker = None
        self.speaker_ids = None
        self.speaker_manager = None
        self.languages = None
        self.new_speaker_ids = None
        self.use_aliases = True

        self.setup_speaker_attributes(speaker_ids_path)

    def setup_speaker_attributes(self, speaker_ids_path):

        # model_inputs = model.get_inputs()
        # use_multi_speaker = len(model_inputs) == 4
        use_multi_speaker = True

        # use_multi_speaker = hasattr(model.tts_model, "num_speakers") and (speaker_ids_path is not None)

        if use_multi_speaker:
            self.new_speaker_ids = json.load(open(speaker_ids_path))

        if self.use_aliases:
            self.speaker_ids = self.new_speaker_ids

        self.languages = ['ca-es', 'ca-ba', 'ca-nw', 'ca-va']

        self.use_multi_speaker = use_multi_speaker
