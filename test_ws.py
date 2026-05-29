import asyncio
import websockets
import json

async def test_ws():
    async with websockets.connect("ws://127.0.0.1:8000/api/ws/call") as websocket:
        await websocket.send(json.dumps({
            "event": "start",
            "lead_name": "TestUser",
            "voice_key": "en_neerja"
        }))
        
        while True:
            try:
                response = await websocket.recv()
                print("Received:", response)
                break
            except Exception as e:
                print("Error receiving:", e)
                break

asyncio.run(test_ws())
