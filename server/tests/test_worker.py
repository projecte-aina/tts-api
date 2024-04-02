from server.helper.config import Config
from server.tests.base_test_case import configBaseTestCase
from server.workers.workers import worker
import pytest


class TestWorker(configBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_worker(self):
        config = Config()
        speaker_config_attributes = config.speakerConfigAttributes.__dict__
        wavs = worker("Es una prova",
                      speaker_id="f_cen_095", 
                      model=config.synthesizer,
                      use_aliases=speaker_config_attributes["use_aliases"],
                      new_speaker_ids=speaker_config_attributes["new_speaker_ids"])
                      

        assert len(wavs) > 1