import os
import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm import load_mission_env


# Map file extensions to correct MIME types for Whisper API
_MIME_MAP = {
    ".webm": "audio/webm",
    ".ogg":  "audio/ogg",
    ".mp3":  "audio/mpeg",
    ".mp4":  "audio/mp4",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
}


class STTService:
    def __init__(self):
        load_mission_env()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "whisper-1"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = None,
    ) -> str:
        """
        Transcribes audio bytes using OpenAI Whisper API.
        - filename must carry the correct extension (.webm/.wav/.ogg) so Whisper
          can auto-detect the codec.
        - language: ISO-639-1 code ('te', 'hi', 'en') improves Indic accuracy.
        """
        if not self.api_key:
            print("Warning: OPENAI_API_KEY is not set.")
            return "[Error: API key missing]"

        if len(audio_bytes) < 1000:
            # Too short to be real speech — avoid wasting an API call
            return ""

        ext = Path(filename).suffix.lower()
        mime = _MIME_MAP.get(ext, "audio/webm")

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Whisper requires the filename extension to be correct for codec detection
        files = {"file": (filename, audio_bytes, mime)}
        data  = {"model": self.model, "response_format": "json"}

        if language:
            data["language"] = language
            
        # CRITICAL: Prime the Whisper model to expect code-mixed Indian languages!
        # This completely eliminates hallucinations (like 'Jennifer Cook') on short audio bursts.
        data["prompt"] = "Hello! Namaste, aap kaise hain? Meeru ela unnaru?"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    try:
                        print(f"  Whisper OK: {text!r}")
                    except UnicodeEncodeError:
                        print("  Whisper OK: [Unicode Text]")
                    return text
                else:
                    print(f"  Whisper error {response.status_code}: {response.text[:200]}")
                    return f"[Error transcribing: {response.status_code}]"

        except httpx.TimeoutException:
            print("  Whisper timeout.")
            return "[Error: Whisper timeout — please speak again]"
        except Exception as e:
            print(f"  Whisper exception: {e}")
            return f"[Error: {str(e)}]"


# Basic test
if __name__ == "__main__":
    import asyncio
    async def test():
        stt = STTService()
        print("Whisper STT service initialized. API Key:", "OK" if stt.api_key else "MISSING")
    asyncio.run(test())
