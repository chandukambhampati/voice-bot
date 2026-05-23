import os
import asyncio
import hashlib
from pathlib import Path
import edge_tts

# Map of friendly keys to Microsoft Edge TTS (Azure Neural) voices
# These voices natively support authentic Indian English, Hindi, and Telugu dialects
VOICE_MAP = {
    "en_neerja":  "en-IN-NeerjaNeural",
    "en_prabhat": "en-IN-PrabhatNeural",
    "hi_swara":   "hi-IN-SwaraNeural",
    "hi_madhur":  "hi-IN-MadhurNeural",
    "te_shruti":  "te-IN-ShrutiNeural",
    "te_mohan":   "te-IN-MohanNeural",
    # Fallback language-code keys
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
}

class TTSService:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent / "static" / "audio"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concurrency limit to prevent overwhelming the Edge TTS server
        self._semaphore = asyncio.Semaphore(4)

    def _get_voice(self, voice_key: str) -> str:
        return VOICE_MAP.get(voice_key, VOICE_MAP.get("en_neerja"))

    def _cache_path(self, text: str, voice: str) -> Path:
        h = hashlib.md5(f"{text}|{voice}".encode("utf-8")).hexdigest()
        return self.output_dir / f"tts_{h}.mp3"

    async def generate_speech(self, text: str, voice_key: str = "en_neerja") -> str:
        """
        Synthesize text → MP3 using Microsoft Edge TTS (Azure Neural Voices).
        This guarantees authentic Indian accents and multi-lingual support.
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
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(str(filepath))

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
        url = await tts.generate_speech("Hello, this is an authentic Indian voice.", "en_neerja")
        print(f"Saved to {url}")

    asyncio.run(test())
