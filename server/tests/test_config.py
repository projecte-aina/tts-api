from server.helper.config import Config
from server.tests.base_test_case import configBaseTestCase
import pytest


class TestConfig(configBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_model_voices(self):
        speaker_ids = ['f_cen_095', 'f_cen_092', 'm_occ_072', 'm_cen_pau', 'm_occ_7b7', 'm_val_51c', 'f_cen_063', 'f_cen_051']
        speaker_config_attributes = Config().speakerConfigAttributes.__dict__
        assert speaker_ids == list(speaker_config_attributes["speaker_ids"].keys())
