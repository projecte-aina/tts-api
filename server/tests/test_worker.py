from server.helper.config import ConfigONNX
from server.tests.base_test_case import configBaseTestCase
from server.workers.workers import worker_onnx_audio_multiaccent
import pytest


class TestWorker(configBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_worker(self):
        config = ConfigONNX()
        speaker_config_attributes = config.speakerConfigAttributes.__dict__
        wavs = worker_onnx_audio_multiaccent(sentence="Es una prova",
                      speaker_id="quim", 
                      model_path="models/matxa_onnx/best_model.onnx",
                      unique_model=True,
                      vocoder_path="models/matxa_onnx/best_model.onnx",
                      use_aliases=speaker_config_attributes["use_aliases"],
                      new_speaker_ids=speaker_config_attributes["new_speaker_ids"],
                      use_cuda=False,
                      temperature=0.4,
                      speaking_rate=1.0)
                      
        assert len(wavs) > 1