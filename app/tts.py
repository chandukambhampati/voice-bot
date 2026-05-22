import os
import asyncio
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Map of friendly keys to OpenAI voices
# OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
VOICE_MAP = {
    # English/Hindi
    "en_neerja":  "nova",     # Female
    "en_prabhat": "echo",     # Male
    "hi_swara":   "nova",     # Female
    "hi_madhur":  "echo",     # Male
    "te_shruti":  "shimmer",  # Female
    "te_mohan":   "onyx",     # Male
    # Fallback language-code keys
    "en": "nova",
    "hi": "nova",
    "te": "shimmer",
}

class TTSService:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent / "static" / "audio"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Safe concurrency limit for OpenAI APIs
        self._semaphore = asyncio.Semaphore(4)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _get_voice(self, voice_key: str) -> str:
        return VOICE_MAP.get(voice_key, VOICE_MAP.get("en_neerja"))

    def _cache_path(self, text: str, voice: str) -> Path:
        h = hashlib.md5(f"{text}|{voice}".encode("utf-8")).hexdigest()
        return self.output_dir / f"tts_{h}.mp3"

    async def generate_speech(self, text: str, voice_key: str = "en_neerja") -> str:
        """
        Synthesize text → MP3 using OpenAI TTS-1 API for low latency.
        """
        text = text.strip()
        if not any(c.isalnum() for c in text):
            return ""

        voice = self._get_voice(voice_key)
        filepath = self._cache_path(text, voice)

        if filepath.exists() and filepath.stat().st_size > 0:
            return f"/static/audio/{filepath.name}"

        async with self._semaphore:
            for attempt in range(3):
                try:
                    response = await self.client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=text
                    )
                    
                    response.stream_to_file(filepath)

                    if filepath.exists() and filepath.stat().st_size > 0:
                        return f"/static/audio/{filepath.name}"
                        
                    print(f"  TTS produced empty file on attempt {attempt+1}")
                except Exception as e:
                    print(f"  TTS error on attempt {attempt+1}: {e}")
                
                await asyncio.sleep(0.5)

            return ""

    async def prewarm(self, voice_key: str = "en_neerja"):
        pass


# CLI test
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    async def test():
        tts = TTSService()
        url = await tts.generate_speech("Hello, this is a test.", "en_neerja")
        print(f"Saved to {url}")

    asyncio.run(test())
