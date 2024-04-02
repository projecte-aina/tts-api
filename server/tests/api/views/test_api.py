import pytest
import json
from fastapi import status
from fastapi.testclient import TestClient

from server.tests.base_test_case import APIBaseTestCase


class TestApi(APIBaseTestCase):
    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.setup()

    def test_text_to_voice(self):
        options = {
            "voice": "f_cen_095",
            "type": "text",
            "text": "hola"
        }
        with TestClient(self.app) as client:
            response = client.post(url="/api/tts", content=json.dumps(options))

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert response.content is not None
 
    def test_text_to_voice_error(self):
        msg = {
            "message":"sfsfs is an unknown speaker id.",
            "accept":["f_cen_095","f_cen_092","m_occ_072","m_cen_pau","m_occ_7b7","m_val_51c","f_cen_063","f_cen_051"]
            }
        options = {
            "voice": "sfsfs",
            "type": "text",
            "text": "hola"
        }
        with TestClient(self.app) as client:
            response = client.post(url="/api/tts", content=json.dumps(options))

        assert response.status_code == status.HTTP_406_NOT_ACCEPTABLE
        content = json.loads(response.content)
        assert content["message"] == msg["message"]
        assert content["accept"] == msg["accept"]

    
    def test_list_voices(self):

        with TestClient(self.app) as client:
            response = client.get(url="/api/available-voices")

        assert response.status_code == status.HTTP_200_OK
        voices = json.loads(response.content)["voices"]
        assert voices == ["f_cen_095","f_cen_092","m_occ_072","m_cen_pau","m_occ_7b7","m_val_51c","f_cen_063","f_cen_051"]