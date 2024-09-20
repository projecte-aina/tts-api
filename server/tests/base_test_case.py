from fastapi.testclient import TestClient
from main import app
from server.helper.config import Config

class APIBaseTestCase:

    def setup(self):
        self.app = app
        self.client = TestClient(self.app)


class configBaseTestCase:
    def setup(self):
        config = Config(
            model_path="models/matxa_onnx/best_model.onnx", 
            config_path="models/vits_ca/config.json", 
            speakers_file_path=None, 
            vocoder_path=None, 
            vocoder_config_path=None, 
            speaker_ids_path="models/matxa_onnx/speaker_ids.json", 
            speech_speed=1.0, 
            use_cuda=False, 
            show_details=True,
            args={}
        )