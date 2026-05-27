import os
import urllib.request
from pathlib import Path

# Base URL for the HuggingFace mirror of Piper voices (which is generally faster/more reliable)
# You can also use the GitHub releases URL if preferred.
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

# List of models to download (both the .onnx and .onnx.json files are required)
MODELS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "hi_IN-swara-medium": "hi/hi_IN/swara/medium/hi_IN-swara-medium",
    "te_IN-shruti-medium": "te/te_IN/shruti/medium/te_IN-shruti-medium"
}

def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded: {dest_path.name}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    # Ensure the models directory exists
    app_dir = Path(__file__).resolve().parent / "app"
    models_dir = app_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {models_dir}")
    
    for model_name, path_suffix in MODELS.items():
        # Download .onnx
        onnx_url = f"{BASE_URL}{path_suffix}.onnx"
        onnx_dest = models_dir / f"{model_name}.onnx"
        if not onnx_dest.exists():
            download_file(onnx_url, onnx_dest)
        else:
            print(f"Already exists: {onnx_dest.name}")
            
        # Download .onnx.json
        json_url = f"{BASE_URL}{path_suffix}.onnx.json"
        json_dest = models_dir / f"{model_name}.onnx.json"
        if not json_dest.exists():
            download_file(json_url, json_dest)
        else:
            print(f"Already exists: {json_dest.name}")

    print("\nAll required Piper TTS models have been verified/downloaded.")

if __name__ == "__main__":
    main()
