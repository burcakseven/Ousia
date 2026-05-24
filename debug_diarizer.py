import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

def debug():
    print("🔍 Debugging Pyannote Diarization Output...")
    token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    
    # Use a small dummy audio or a tiny slice if available, 
    # but here we can just check the type and attributes of a mock call or the real thing.
    audio_path = "samples/ramani_5min.wav"
    if not os.path.exists(audio_path):
        print("Audio not found.")
        return

    output = pipeline(audio_path)
    print(f"Type: {type(output)}")
    print(f"Attributes: {dir(output)}")
    
    if hasattr(output, "itertracks"):
        print("Has itertracks!")
    
    # Check common internal names
    for attr in ["annotation", "segments", "labels", "tracks"]:
        if hasattr(output, attr):
            print(f"Has attribute: {attr}")

if __name__ == "__main__":
    debug()
