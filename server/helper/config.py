# from argparse import Namespace
import json
# from TTS.utils.synthesizer import Synthesizer

from server.helper.singleton import Singleton
from scripts.inference_onnx import load_onnx_tts

'''
class Config(metaclass=Singleton):
    def __init__(self, model_path, config_path, speakers_file_path, 
                 vocoder_path, vocoder_config_path, speaker_ids_path, 
                 speech_speed, mp_workers, use_cuda, use_mp, show_details, args) -> None:
        self.speech_speed = speech_speed
        self.mp_workers = mp_workers
        self.use_cuda = use_cuda
        self.use_mp = use_mp
        self.config_path = config_path
        self.vocoder_config_path = vocoder_config_path
        self.show_details = show_details
        self.args = args

        self.synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=speakers_file_path,
                tts_languages_file=None,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config_path,
                encoder_checkpoint="",
                encoder_config="",
                use_cuda=use_cuda
            )

        self.speakerConfigAttributes = SpeakerConfigAttributes(self.synthesizer, speaker_ids_path)
'''


class ConfigONNX(metaclass=Singleton):
    def __init__(self, model_path, vocoder_path, speaker_ids_path,
                 speech_speed, temperature, mp_workers, use_cuda, use_mp, unique_model) -> None:
        self.speech_speed = speech_speed
        self.temperature = temperature
        self.mp_workers = mp_workers
        self.use_cuda = use_cuda
        self.use_mp = use_mp
        self.model_path = model_path
        self.vocoder_path = vocoder_path
        self.unique_model = unique_model

        # self.model_tts, self.vocoder = load_onnx_tts(model_path=model_path, vocoder_path=vocoder_path, use_cuda=False)

        # speakers_id_path es el JSON con los nombres de los speakers
        self.speakerConfigAttributes = SpeakerConfigAttributes(speaker_ids_path)


'''
class SpeakerConfigAttributes:
    def __init__(self, synthesizer, speaker_ids_path) -> None:
        self.use_multi_speaker = None
        self.speaker_ids = None
        self.speaker_manager = None
        self.languages = None
        self.new_speaker_ids = None
        self.use_aliases = True
        self.use_gst = None

        self.setup_speaker_attributes(synthesizer, speaker_ids_path)

    def setup_speaker_attributes(self, model, speaker_ids_path):
        # global new_speaker_ids, use_aliases

        use_multi_speaker = hasattr(model.tts_model, "num_speakers") and (
            model.tts_model.num_speakers > 1 or model.tts_speakers_file is not None)

        speaker_manager = getattr(model.tts_model, "speaker_manager", None)
        if speaker_manager:
            self.new_speaker_ids = json.load(open(speaker_ids_path))

        if self.use_aliases:
            self.speaker_ids = self.new_speaker_ids
        else:
            self.speaker_ids = speaker_manager.ids

        self.languages = ['ca-es']

        # TODO: set this from SpeakerManager
        self.use_gst = model.tts_config.get("use_gst", False)

        self.use_multi_speaker = use_multi_speaker
        self.speaker_manager = speaker_manager
'''


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

        self.languages = ['ca-es']

        self.use_multi_speaker = use_multi_speaker
