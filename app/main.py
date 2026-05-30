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
from app.emotion import EmotionService

app = FastAPI(title="Oncology Voice Bot App - Dr. Bharat Patodiya")

# Define static directories
app_dir = Path(__file__).resolve().parent
static_dir = app_dir / "static"
audio_dir = static_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize services globally but handle exceptions so Uvicorn boots successfully
agent = None
tts_service = None
stt_service = None
emotion_service = None
init_error = None
try:
    agent = OncologyAgent()
    tts_service = TTSService(output_dir=str(audio_dir))
    stt_service = STTService()
    emotion_service = EmotionService()
except Exception as e:
    print(f"FATAL INITIALIZATION ERROR: {e}")
    init_error = traceback.format_exc()
    traceback.print_exc()

# In-memory session store
sessions = {}

@app.on_event("startup")
async def startup_prewarm():
    """Pre-warm TTS voices so the first caller doesn't wait for cold-start synthesis."""
    import asyncio
    if tts_service:
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
    if not agent:
        return {"error": "Server started but AI Services failed to initialize.", "traceback": init_error}
    
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
    
    # Track the active agent task for interruption
    active_tasks = {}
    
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
                initial_greeting = f"Namaste andi, meeru {lead_name} garena matladedi?"
            else:
                initial_greeting = "Namaste andi! Advanced Cancer Center nundi Dr. Bharat Patodiya gari tarapuna Ankur matladutunnanandi. Nenu evaritho matladutunnano telusukovachha andi?"
        elif voice_key.startswith("hi"): # Hindi
            if lead_name:
                initial_greeting = f"Namaste, kya main {lead_name} ji se baat kar raha hoon?"
            else:
                initial_greeting = "Namaste! Main Advanced Cancer Center se Doctor Bharat Patodiya ji ki taraf se Ankur bol raha hoon. Kya main aapka shubh naam jaan sakta hoon?"
        else: # English
            if lead_name:
                initial_greeting = f"Hello, is this Mr. {lead_name}?"
            else:
                initial_greeting = "Hello! Ankur here from Advanced Cancer Center on behalf of Dr. Bharat Patodiya. May I know who I am speaking with, please?"
                
        sessions[session_id]["history"].append({"role": "assistant", "content": initial_greeting})
        
        # Generate and send the initial greeting audio
        try:
            await websocket.send_json({
                "event": "greeting",
                "text": initial_greeting,
                "stage": "G",
                "tags": sessions[session_id]["tags"]
            })
            
            async for chunk in tts_service.generate_speech(initial_greeting, voice_key=voice_key):
                if chunk.startswith(b'{"event":'):
                    await websocket.send_text(chunk.decode('utf-8'))
                else:
                    await websocket.send_bytes(chunk)
                
        except Exception as e:
            print(f"[{session_id}] Greeting TTS error: {e}")
        
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

                if event == "interrupt":
                    print(f"[{session_id}] Received INTERRUPT signal. Cancelling active tasks.")
                    if session_id in active_tasks:
                        active_tasks[session_id].cancel()
                        del active_tasks[session_id]
                    continue

                if event == "text_input":
                    user_text = message.get("text", "").strip()
                    voice_key = message.get("voice_key", "en_neerja")
                    if user_text:
                        if session_id in active_tasks:
                            active_tasks[session_id].cancel()
                        task = asyncio.create_task(process_agent_turn(websocket, session_id, user_text, voice_key, "auto", 150))
                        active_tasks[session_id] = task

                elif event == "audio_input":
                    audio_b64   = message.get("audio", "")
                    voice_key   = message.get("voice_key", "en_neerja")
                    lang_hint   = message.get("language_hint", None)

                    if not audio_b64:
                        continue

                    # Immediate ACK — tells the UI to show "Thinking..." right away
                    await websocket.send_json({"event": "processing"})

                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Detect emotion from audio
                    detected_emotion = "calm"
                    try:
                        detected_emotion = await emotion_service.detect_emotion(audio_bytes)
                        print(f"[{session_id}] Detected emotion: {detected_emotion}")
                    except Exception as e:
                        print(f"[{session_id}] Emotion detection error: {e}")

                    # Use client-provided filename (carries correct extension for MIME detection)
                    # Fall back to .webm for Indic, .wav for English
                    fname = message.get("filename") or (
                        "input_audio.webm" if voice_key.startswith(("te", "hi")) else "input_audio.wav"
                    )
                    print(f"[{session_id}] STT start — lang={lang_hint}, file={fname}, size={len(audio_bytes)}B")
                    try:
                        user_text, wpm = await stt_service.transcribe_audio(
                            audio_bytes, filename=fname, language=lang_hint
                        )
                        print(f"[{session_id}] OpenAI STT transcript: {user_text!r}")
                    except Exception as e:
                        print(f"[{session_id}] STT error: {e}")
                        user_text, wpm = f"[Error: {e}]", 0

                    if user_text.strip() and not user_text.startswith("[Error"):
                        if session_id in active_tasks:
                            active_tasks[session_id].cancel()
                        task = asyncio.create_task(process_agent_turn(websocket, session_id, user_text, voice_key, detected_emotion, wpm))
                        active_tasks[session_id] = task
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

async def process_agent_turn(websocket: WebSocket, session_id: str, user_text: str, voice_key: str, emotion: str = "calm", wpm: int = 150):
    global sessions
    state = sessions[session_id]
    
    # Send processing status
    await websocket.send_json({"event": "processing"})
    print(f"[{session_id}] Agent streaming response...")
    
    await websocket.send_json({"event": "stream_start"})
    
    current_sentence_buffer = ""
    sentence_count = 0
    
    # Audio Pipeline
    sentence_queue = asyncio.Queue()
    
    async def tts_consumer():
        try:
            while True:
                item = await sentence_queue.get()
                if item is None:
                    break
                sentence, vk, dyn_emotion, w_pm = item
                
                async for chunk in tts_service.generate_speech(sentence, voice_key=vk, target_wpm=w_pm, emotion=dyn_emotion):
                    if asyncio.current_task().cancelled():
                        return
                    if chunk.startswith(b'{"event":'):
                        await websocket.send_text(chunk.decode("utf-8"))
                    else:
                        await websocket.send_bytes(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"tts_consumer error: {e}")
            
    tts_task = asyncio.create_task(tts_consumer())
    
    sentence_end_regex = re.compile(r'([.?!।\n]\s*)')
    try:
        full_response_text = ""
        current_sentence = ""
        dynamic_voice_key = voice_key
        dynamic_emotion = emotion  # Default to whatever was detected acoustically or "calm"
        buffer = ""
        tag_extracted = False

        async for chunk in agent.stream_turn(user_text, state["history"], state["stage"], state["tags"], emotion, wpm):
            if chunk["event"] == "citations":
                await websocket.send_json(chunk)
            elif chunk["event"] == "text_chunk":
                t = chunk["text"]
                
                if not tag_extracted:
                    buffer += t
                    # Need to wait until we have enough buffer to see if tags are present
                    if "]" in buffer:
                        # Find all tags like [LANG:XX] or [EMOTION:YY]
                        lang_match = re.search(r'\[LANG:(EN|HI|TE)\]', buffer, re.IGNORECASE)
                        emotion_match = re.search(r'\[EMOTION:(sad|angry|happy|calm)\]', buffer, re.IGNORECASE)
                        
                        if lang_match:
                            lang = lang_match.group(1).upper()
                            if lang == 'TE': dynamic_voice_key = "te_shruti"
                            elif lang == 'HI': dynamic_voice_key = "hi_swara"
                            elif lang == 'EN': dynamic_voice_key = "en_neerja"
                            
                        if emotion_match:
                            dynamic_emotion = emotion_match.group(1).lower()
                            print(f"[{session_id}] LLM elected emotion: {dynamic_emotion}")
                        
                        # Once we see a closing bracket, strip all tags and output the rest
                        t = re.sub(r'\[.*?\]', '', buffer).lstrip()
                        tag_extracted = True
                    elif len(buffer) > 25 and "[" not in buffer:
                        # If buffer is getting long and there's no open bracket, assume no tags
                        tag_extracted = True
                        t = buffer
                    elif len(buffer) > 40:
                        # Failsafe if the bracket is malformed
                        tag_extracted = True
                        t = buffer
                    else:
                        continue # wait for more characters to find the tag

                if t:
                    full_response_text += t
                    current_sentence += t
                    await websocket.send_json({"event": "text_chunk", "text": t})
                    
                    if any(p in current_sentence for p in ['. ', '? ', '! ', ', ', '; ', '\n']):
                        sentences = re.split(r'(?<=[.?!,;])\s+|\n+', current_sentence)
                        complete_sentence = sentences[0].strip()
                        if complete_sentence:
                            await sentence_queue.put((complete_sentence, dynamic_voice_key, dynamic_emotion, wpm))
                            current_sentence = ' '.join(sentences[1:])
            
            if asyncio.current_task().cancelled():
                tts_task.cancel()
                return

    except Exception as e:
        print(f"[{session_id}] agent turn error: {e}")
        traceback.print_exc()
    finally:
        if asyncio.current_task().cancelled():
            print(f"[{session_id}] Turn cancelled. Stopping TTS queue.")
            tts_task.cancel()
        else:
            # Push any remaining text as the last sentence
            if current_sentence.strip():
                await sentence_queue.put((current_sentence.strip(), dynamic_voice_key, dynamic_emotion, wpm))
            
            # Signal consumer to stop and wait for it
            await sentence_queue.put(None)
            await tts_task

        state["history"].append({"role": "assistant", "content": full_response_text})

    # 5. Finalize conversational turn immediately so frontend mic unlocks
    await websocket.send_json({
        "event": "stream_end",
        "stage": state["stage"],
        "tags": state["tags"]
    })
    print(f"[{session_id}] Streaming complete. Voice unlocked.")

    # 6. Extract CRM tags entirely in the background so it never blocks the UI
    async def background_update_tags():
        try:
            state_result = await agent.update_state(state["history"], state["stage"], state["tags"])
            new_stage = state_result.get("call_stage", state["stage"])
            new_tags = state_result.get("crm_tags", state["tags"])
            
            state["stage"] = new_stage
            state["tags"] = new_tags
            
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
