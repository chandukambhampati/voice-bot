import requests
import json
import base64

url = "https://api.sarvam.ai/text-to-speech"
headers = {
    "api-subscription-key": "sk_wldri34t_eXQzwO5yt4N5Y95X10MKBaBh",
    "Content-Type": "application/json"
}
payload = {
    "inputs": ["Hello, how are you?"],
    "target_language_code": "hi-IN",
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.0,
    "loudness": 1.5,
    "speech_sample_rate": 16000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
}
try:
    response = requests.post(url, headers=headers, json=payload)
    print("bulbul:v1 status:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("Keys in response:", data.keys())
        if "audios" in data:
            print("Audio length:", len(data["audios"][0]))
    else:
        print("Error:", response.text)
except Exception as e:
    print(e)

# Also test bulbul:v3 syntax from the search results
payload_v3 = {
    "text": "Hello, welcome.",
    "target_language_code": "en-IN",
    "model": "bulbul:v3"
}
try:
    response = requests.post(url, headers=headers, json=payload_v3)
    print("bulbul:v3 status:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("Keys in v3 response:", data.keys())
        # Let's see what's in the response
        if "audio" in data:
            print("v3 Audio length:", len(data["audio"]))
    else:
        print("v3 Error:", response.text)
except Exception as e:
    print(e)
