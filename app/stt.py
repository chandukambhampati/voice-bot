import os
import sys
import io
import time
from pathlib import Path
from openai import AsyncOpenAI
from abc import ABC, abstractmethod

sys.path.append(str(Path(__file__).resolve().parents[1]))

class BaseSTTProvider(ABC):
    @abstractmethod
    async def transcribe_audio(self, audio_bytes: bytes, filename: str, language: str = None) -> tuple[str, int]:
        pass

class OpenAISTTProvider(BaseSTTProvider):
    def __init__(self):
        # Pass dummy key if missing to prevent instant crash on container startup
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key")) 
        print("Provider: OpenAI Whisper STT initialized.")

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = "hi", # default to Hindi for Indic context
    ) -> tuple[str, int]:
        """
        Transcribes audio bytes using local faster-whisper.
        Returns a tuple: (transcript_text, words_per_minute)
        """
        if len(audio_bytes) < 1000:
            return "", 0

        try:
            start_time = time.time()
            # OpenAI requires a filename to determine the format. We'll pass it as a tuple.
            file_tuple = (filename, audio_bytes, "audio/webm" if "webm" in filename else "audio/wav")
            
            # Request verbose_json to get segment durations to calculate WPM
            response = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=file_tuple,
                response_format="verbose_json"
            )
            
            transcript = response.text.strip()
            word_count = len(transcript.split())
            audio_duration = response.duration if hasattr(response, 'duration') and response.duration else 0.0
            
            wpm = 0
            if audio_duration > 0:
                wpm = int((word_count / audio_duration) * 60)
            
            print(f"  OpenAI Whisper OK: {transcript!r} (WPM: {wpm}) in {time.time() - start_time:.2f}s")
            return transcript, wpm
            
        except Exception as e:
            import traceback
            print(f"  OpenAI STT exception: {e}")
            traceback.print_exc()
            return f"[Error: {str(e)}]", 0

class STTService:
    def __init__(self):
        # We can add SarvamSTTProvider here later if needed
        self.provider = OpenAISTTProvider()

    async def transcribe_audio(self, audio_bytes: bytes, filename: str, language: str = None) -> tuple[str, int]:
        return await self.provider.transcribe_audio(audio_bytes, filename, language)

# Basic test
if __name__ == "__main__":
    import asyncio
    async def test():
        stt = STTService()
        print("Local STT service initialized.")
    asyncio.run(test())
