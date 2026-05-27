import os
import io
import wave
try:
    from transformers import pipeline
    import torchaudio
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/torchaudio not installed. Falling back to mock Emotion Detection.")

class EmotionService:
    def __init__(self):
        self.classifier = None
        if HAS_TRANSFORMERS:
            print("Loading Speech Emotion Recognition (SER) model...")
            # Using a standard lightweight emotion recognition model from Hugging Face
            # In a real production scenario with GPUs, you might load this on 'cuda'
            self.classifier = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device=-1 # CPU by default for broader compatibility
            )
            print("SER model loaded.")

    async def detect_emotion(self, audio_bytes: bytes) -> str:
        """
        Takes raw audio bytes (expected to be WAV or converted to WAV) 
        and returns the dominant emotion.
        """
        if len(audio_bytes) < 1000:
            return "neutral"

        if not HAS_TRANSFORMERS or not self.classifier:
            print("  [Mock SER] Detecting emotion...")
            return "calm"

        try:
            # The pipeline expects a path or raw waveform. 
            # For simplicity, we assume we can pass the raw bytes if they are valid wav,
            # or we might need to decode them. The pipeline often handles raw dicts of {"array": ..., "sampling_rate": ...}
            # For this implementation, we will pass the raw bytes and let the pipeline decode it if supported,
            # or gracefully fallback to neutral.
            
            # In a robust implementation, you'd use ffmpeg to ensure it's a 16kHz mono WAV first.
            results = self.classifier(audio_bytes)
            
            if results and isinstance(results, list):
                # The pipeline returns a list of dicts: [{'score': 0.9, 'label': 'hap'}, ...]
                # superb/wav2vec2-base-superb-er labels: 'neu', 'hap', 'ang', 'sad'
                top_emotion = results[0]['label']
                
                # Map to readable format
                emotion_map = {
                    "neu": "calm",
                    "hap": "happy",
                    "ang": "angry",
                    "sad": "sad"
                }
                return emotion_map.get(top_emotion, "calm")
                
        except Exception as e:
            print(f"Emotion detection error: {e}")
            
        return "calm"
