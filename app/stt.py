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

class OpenAIWhisperSTTProvider(BaseSTTProvider):
    def __init__(self):
        print("Provider: Local OpenAI-Whisper STT initialized (100% Free).")
        try:
            import whisper
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = whisper.load_model("tiny", device="cpu")
        except Exception as e:
            print(f"Failed to load OpenAI Whisper: {e}")
            self.model = None

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        language: str = None
    ) -> tuple[str, int]:
        if len(audio_bytes) < 1000 or not self.model:
            return "", 0

        try:
            start_time = time.time()
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            import asyncio
            def _transcribe():
                kwargs = {"fp16": False}
                if language:
                    kwargs["language"] = language
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.model.transcribe(tmp_path, **kwargs)
                return result["text"], result.get("segments", [])

            full_text, segments = await asyncio.to_thread(_transcribe)
            
            try:
                os.remove(tmp_path)
            except:
                pass
            
            duration = sum([s["end"] - s["start"] for s in segments]) if segments else 1.0
            wpm = 0
            if duration > 0:
                wpm = int((len(full_text.split()) / duration) * 60)
            
            print(f"  Local STT OK: {full_text!r} (WPM: {wpm}) in {time.time() - start_time:.2f}s")
            return full_text, wpm
            
        except Exception as e:
            import traceback
            print(f"  Local STT exception: {e}")
            traceback.print_exc()
            return f"[Error: {str(e)}]", 0

class STTService:
    def __init__(self):
        self.provider = OpenAIWhisperSTTProvider()

    async def transcribe_audio(self, audio_bytes: bytes, filename: str, language: str = None) -> tuple[str, int]:
        # For true auto-detect on a real phone call, we completely ignore frontend language hints!
        return await self.provider.transcribe_audio(audio_bytes, filename, language=None)

# Basic test
if __name__ == "__main__":
    import asyncio
    async def test():
        stt = STTService()
        print("Local STT service initialized.")
    asyncio.run(test())
