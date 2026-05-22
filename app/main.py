import os
import json
import asyncio
import base64
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.agent import OncologyAgent
from app.tts import TTSService
from app.stt import STTService

app = FastAPI(title="Oncology Voice Bot App - Dr. Bharat Patodiya")

# Define static directories
app_dir = Path(__file__).resolve().parent
static_dir = app_dir / "static"
audio_dir = static_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize services
agent = OncologyAgent()
tts_service = TTSService(output_dir=str(audio_dir))
stt_service = STTService()

# In-memory session store
sessions = {}

@app.on_event("startup")
async def startup_prewarm():
    """Pre-warm TTS voices so the first caller doesn't wait for cold-start synthesis."""
    import asyncio
    print("Prewarming TTS voices...")
    await asyncio.gather(
        tts_service.prewarm("en_neerja"),
        tts_service.prewarm("hi_swara"),
        tts_service.prewarm("te_shruti"),
    )
    print("TTS prewarm complete.")

@app.get("/")
async def get_index():
    """Serves the main frontend dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return {"message": "Voice Bot Backend Running. Please create index.html in app/static."}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(static_dir / "favicon.ico") if (static_dir / "favicon.ico").exists() else {}

# ---------------------------------------------------------------------------
# Browser WebSocket Endpoint
# ---------------------------------------------------------------------------
@app.websocket("/api/ws/call")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = f"session_{id(websocket)}"
    sessions[session_id] = {
        "stage": "G",
        "tags": {
            "caller_name": "Unsure",
            "patient_relation": "Unsure",
            "cancer_type": "Unsure",
            "symptoms": [],
            "decision_authority": "Unsure",
            "travel_fit": "Unsure",
            "lead_temperature": "Warm",
            "buyer_persona": "Unsure",
            "consultation_booked": "None"
        },
        "history": []
    }
    
    print(f"New browser connection: {session_id}")
    
    try:
        # Wait for the first message (start/config) from client
        data = await websocket.receive_text()
        message = json.loads(data)
        
        event = message.get("event")
        lead_name = message.get("lead_name", "").strip()
        voice_key = message.get("voice_key", "en_neerja")
        
        # Initialize session tags with the lead_name if provided
        caller_name = lead_name if lead_name else "Unsure"
        sessions[session_id]["tags"]["caller_name"] = caller_name
        
        # Determine greeting language and phrasing based on voice_key and lead_name
        if voice_key.startswith("te"): # Telugu
            if lead_name:
                initial_greeting = f"నమస్తే అండి, మీరు {lead_name} గారేనా మాట్లాడేది?"
            else:
                initial_greeting = "నమస్తే అండి! అడ్వాన్స్డ్ క్యాన్సర్ సెంటర్ నుండి డాక్టర్ భరత్ పటోడియా గారి తరపున అంకుర్ మాట్లాడుతున్నానండి. నేను ఎవరితో మాట్లాడుతున్నానో తెలుసుకోవచ్చా అండి?"
        elif voice_key.startswith("hi"): # Hindi
            if lead_name:
                initial_greeting = f"नमस्ते, क्या मैं {lead_name} जी से बात कर रहा हूँ?"
            else:
                initial_greeting = "नमस्ते! मैं एडवांस्ड कैंसर सेंटर से डॉक्टर भरत पटोदिया जी की तरफ से अंकुर बोल रहा हूँ। क्या मैं आपका शुभ नाम जान सकता हूँ?"
        else: # English
            if lead_name:
                initial_greeting = f"Hello, is this Mr. {lead_name}?"
            else:
                initial_greeting = "Hello! Ankur here from Advanced Cancer Center on behalf of Dr. Bharat Patodiya. May I know who I am speaking with, please?"
                
        sessions[session_id]["history"].append({"role": "assistant", "content": initial_greeting})
        
        # Generate initial TTS
        audio_url = await tts_service.generate_speech(initial_greeting, voice_key=voice_key)
        
        await websocket.send_json({
            "event": "greeting",
            "text": initial_greeting,
            "audio_url": audio_url,
            "stage": "G",
            "tags": sessions[session_id]["tags"]
        })
        
        # 2. Main communication loop — per-message error isolation prevents single bad
        #    packets from killing the entire WebSocket connection.
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"Browser disconnected: {session_id}")
                break

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                print(f"Malformed JSON from {session_id}, skipping.")
                continue

            try:
                event = message.get("event")

                if event == "text_input":
                    user_text = message.get("text", "").strip()
                    voice_key = message.get("voice_key", "en_neerja")
                    if user_text:
                        await process_agent_turn(websocket, session_id, user_text, voice_key)

                elif event == "audio_input":
                    audio_b64   = message.get("audio", "")
                    voice_key   = message.get("voice_key", "en_neerja")
                    lang_hint   = message.get("language_hint", None)

                    if not audio_b64:
                        continue

                    # Immediate ACK — tells the UI to show "Thinking..." right away
                    await websocket.send_json({"event": "processing"})

                    audio_bytes = base64.b64decode(audio_b64)

                    # Use client-provided filename (carries correct extension for MIME detection)
                    # Fall back to .webm for Indic, .wav for English
                    fname = message.get("filename") or (
                        "input_audio.webm" if voice_key.startswith(("te", "hi")) else "input_audio.wav"
                    )
                    print(f"[{session_id}] STT start — lang={lang_hint}, file={fname}, size={len(audio_bytes)}B")
                    user_text = await stt_service.transcribe_audio(
                        audio_bytes, filename=fname, language=lang_hint
                    )
                    print(f"[{session_id}] Whisper transcript: {user_text!r}")

                    if user_text.strip() and not user_text.startswith("[Error"):
                        await process_agent_turn(websocket, session_id, user_text, voice_key)
                    else:
                        err = user_text if user_text.startswith("[Error") else ""
                        await websocket.send_json({"event": "stt_error", "error": err})

            except WebSocketDisconnect:
                print(f"Browser disconnected mid-message: {session_id}")
                break
            except Exception as e:
                # Log but do NOT re-raise — keeps connection alive
                print(f"[{session_id}] Error processing message: {e}")
                traceback.print_exc()
                try:
                    await websocket.send_json({"event": "stt_error", "error": ""})
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"Browser disconnected (setup): {session_id}")
    except Exception as e:
        print(f"WebSocket setup error in {session_id}: {e}")
        traceback.print_exc()
    finally:
        if session_id in sessions:
            del sessions[session_id]


import re

async def process_agent_turn(websocket: WebSocket, session_id: str, user_text: str, voice_key: str):
    """Process one agent turn using a fast, streaming pipeline."""
    session = sessions[session_id]

    # 1. Append user message
    session["history"].append({"role": "user", "content": user_text})

    # 2. Echo transcript + indicate start
    await websocket.send_json({"event": "user_transcript", "text": user_text})
    await websocket.send_json({"event": "stream_start"})

    full_response_text = ""
    current_sentence_buffer = ""
    sentence_count = 0
    citations = []

    # Regex to split on full sentence boundaries ONLY
    # (Generating audio piece-by-piece ruins prosody/intonation. Full sentences sound much more human)
    sentence_end_regex = re.compile(r'([.?!।\n]\s*)')

    # Background tasks for TTS generation
    tts_tasks = []

    print(f"[{session_id}] Agent streaming response...")

    # 3. Process LLM token stream
    async for stream_event in agent.stream_turn(user_text, session["history"], session["stage"], session["tags"]):
        if stream_event["event"] == "citations":
            citations = stream_event["citations"]
            if citations:
                await websocket.send_json({"event": "citations", "citations": citations})
        
        elif stream_event["event"] == "text_chunk":
            chunk = stream_event["text"]
            full_response_text += chunk
            current_sentence_buffer += chunk

            # Send raw text chunk for "live typing" feel in UI
            await websocket.send_json({"event": "text_chunk", "text": chunk})

            # Check if buffer contains a complete sentence
            match = sentence_end_regex.search(current_sentence_buffer)
            if match:
                end_pos = match.end()
                complete_sentence = current_sentence_buffer[:end_pos].strip()
                current_sentence_buffer = current_sentence_buffer[end_pos:]

                if complete_sentence:
                    sentence_count += 1
                    # Fire off TTS generation in the background immediately
                    idx = sentence_count
                    task = asyncio.create_task(
                        tts_service.generate_speech(complete_sentence, voice_key=voice_key)
                    )
                    tts_tasks.append((idx, complete_sentence, task))

    # Handle any remaining text in buffer
    if current_sentence_buffer.strip():
        sentence_count += 1
        idx = sentence_count
        complete_sentence = current_sentence_buffer.strip()
        task = asyncio.create_task(
            tts_service.generate_speech(complete_sentence, voice_key=voice_key)
        )
        tts_tasks.append((idx, complete_sentence, task))

    # 4. Stream generated audio chunks as they finish
    # We await them in order so the audio plays sequentially
    for idx, sentence_text, task in tts_tasks:
        try:
            audio_url = await task
            if audio_url:
                await websocket.send_json({
                    "event": "audio_sentence",
                    "audio_url": audio_url,
                    "sentence_idx": idx,
                    "is_last": (idx == sentence_count)
                })
        except Exception as e:
            print(f"[{session_id}] TTS chunk error: {e}")

    session["history"].append({"role": "assistant", "content": full_response_text})

    # 5. Finalize conversational turn immediately so frontend mic unlocks
    await websocket.send_json({
        "event": "stream_end",
        "stage": session["stage"],
        "tags": session["tags"]
    })
    print(f"[{session_id}] Streaming complete. Voice unlocked.")

    # 6. Extract CRM tags entirely in the background so it never blocks the UI
    async def background_update_tags():
        try:
            state_result = await agent.update_state(session["history"], session["stage"], session["tags"])
            new_stage = state_result.get("call_stage", session["stage"])
            new_tags = state_result.get("crm_tags", session["tags"])
            
            session["stage"] = new_stage
            session["tags"] = new_tags
            
            # Send stealth update to UI
            await websocket.send_json({
                "event": "tags_update",
                "stage": new_stage,
                "tags": new_tags
            })
        except Exception as e:
            print(f"[{session_id}] CRM Tag extraction failed: {e}")

    asyncio.create_task(background_update_tags())

# ---------------------------------------------------------------------------
# Twilio Integration Endpoints (Mocked for Telephony Documentation)
# ---------------------------------------------------------------------------
@app.post("/api/twilio/voice")
async def twilio_voice(request: Request):
    """
    TwiML Webhook endpoint called by Twilio when a call is received.
    Directs Twilio to stream audio over WebSockets to our `/api/twilio/stream` endpoint.
    """
    host = request.headers.get("host", "your-server-url.ngrok-free.app")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">Connecting to Advanced Cancer Center assistant...</Say>
    <Connect>
        <Stream url="{scheme}://{host}/api/twilio/stream" />
    </Connect>
</Response>
"""
    return HTMLResponse(content=twiml_response, media_type="application/xml")

@app.websocket("/api/twilio/stream")
async def twilio_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Twilio streams telephony audio in µ-law 8000Hz format, base64-encoded in JSON.
    """
    await websocket.accept()
    print("Twilio Media Stream WebSocket Connected")
    
    # Set up session similar to browser
    session_id = f"twilio_{id(websocket)}"
    sessions[session_id] = {
        "stage": "G",
        "tags": {
            "caller_name": "Unsure",
            "patient_relation": "Unsure",
            "cancer_type": "Unsure",
            "symptoms": [],
            "decision_authority": "Unsure",
            "travel_fit": "Unsure",
            "lead_temperature": "Warm",
            "buyer_persona": "Unsure",
            "consultation_booked": "None"
        },
        "history": []
    }
    
    try:
        # In a real telephony server, we receive audio packet-by-packet,
        # detect silence (VAD), run Whisper STT, run Agent, and stream TTS audio back.
        # This loop demonstrates how the JSON stream events are parsed.
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)
            
            event = packet.get("event")
            
            if event == "start":
                print(f"Twilio Call Started: StreamSid {packet.get('streamSid')}")
                # Send greeting
                initial_greeting = "Hello, is this Mr. Sameer?"
                sessions[session_id]["history"].append({"role": "assistant", "content": initial_greeting})
                
                # Render TTS as µ-law (in production, we'd encode to ulaw 8k; here we show outline)
                # tts_ulaw = await tts_service.generate_speech_ulaw(initial_greeting, voice="hi-IN-SwaraNeural")
                # await websocket.send_json({
                #     "event": "media",
                #     "media": {"payload": base64.b64encode(tts_ulaw).decode('utf-8')}
                # })
                
            elif event == "media":
                # Realtime audio streaming chunk from user
                payload = packet.get("media", {}).get("payload", "")
                # Here we would feed audio to a Voice Activity Detector (VAD)
                # and transcribe when user stops speaking.
                pass
                
            elif event == "stop":
                print(f"Twilio Call Stopped: {session_id}")
                break
                
    except WebSocketDisconnect:
        print(f"Twilio disconnected: {session_id}")
    finally:
        if session_id in sessions:
            del sessions[session_id]
