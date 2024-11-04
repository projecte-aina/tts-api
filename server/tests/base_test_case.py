from fastapi.testclient import TestClient
from main import app
from server.helper.config import ConfigONNX

class APIBaseTestCase:

    def setup(self):
        self.app = app
        self.client = TestClient(self.app)


class configBaseTestCase:
    def setup(self):
        config = ConfigONNX(
            model_path="models/matxa_onnx/best_model.onnx", 
            vocoder_path="models/matxa_onnx/best_model.onnx",  
            speaker_ids_path="models/matxa_onnx/spk_ids.json", 
            temperature=0.4,
            speech_speed=1.0, 
            use_cuda=False, 
            unique_model=True,
        )
