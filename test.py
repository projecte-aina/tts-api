import httpx
translate_service_url = "http://127.0.0.1:8000/api/tts"
json_data ={
    "voice": "f_cen_81",
    "text": "Aquell vint-i-tres de maig de mil dos-cents vuitanta-set, Sant Cristau va senyalar amb una foguera l’aparició d’una tropa enemiga a la plana de l’Alt Empordà.",
    "type": "text"
}
r = httpx.post(translate_service_url, json=json_data, timeout=3000)
response = r.json()
print("Translation:", response)