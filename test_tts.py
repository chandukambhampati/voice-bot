import asyncio
import os
import json
import websockets
from dotenv import load_dotenv

load_dotenv()

async def test():
    uri = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3"
    api_key = os.environ.get("SARVAM_API_KEY")
    headers = {"api-subscription-key": api_key}
    
    print("Connecting...")
    async with websockets.connect(uri, additional_headers=headers) as ws:
        config_msg = {
            "type": "config",
            "data": {
                "target_language_code": "hi-IN",
                "speaker": "priya"
            }
        }
        await ws.send(json.dumps(config_msg))
        print("Config sent.")
        
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"Config Response: {response}")
        except asyncio.TimeoutError:
            print("No error after config. Config is likely valid. Now sending text...")
            
            text_msg = {
                "type": "text",
                "data": {
                    "text": "Hello, how are you today?"
                }
            }
            await ws.send(json.dumps(text_msg))
            flush_msg = {"type": "flush", "data": {}}
            await ws.send(json.dumps(flush_msg))
            
            import base64
            with open("test_ws_out.raw", "wb") as f:
                while True:
                    response = await ws.recv()
                    res_json = json.loads(response)
                    msg_type = res_json.get("type")
                    if msg_type == "audio":
                        audio_b64 = res_json.get("data", {}).get("audio", "")
                        if audio_b64:
                            f.write(base64.b64decode(audio_b64))
                            print(f"Got chunk: {len(audio_b64)} b64 chars")
                        else:
                            print(f"Empty audio data: {res_json}")
                    else:
                        print(f"Other Message: {res_json}")
                        if msg_type == "error":
                            break

if __name__ == "__main__":
    asyncio.run(test())
