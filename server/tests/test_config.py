from server.helper.config import ConfigONNX
from server.tests.base_test_case import configBaseTestCase
import pytest


class TestConfig(configBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_model_voices(self):
        speaker_ids = ["quim","olga","grau","elia","pere","emma","lluc","gina"]
        speaker_config_attributes = ConfigONNX().speakerConfigAttributes.__dict__
        assert speaker_ids == list(speaker_config_attributes["speaker_ids"].keys())
