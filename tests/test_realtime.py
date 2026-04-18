"""
Test real-time streaming pipeline.

Usage:
    python test_realtime.py            # streams sample_speech.wav as chunks
    python test_realtime.py --mic      # live mic (requires sounddevice)
"""

import argparse
import numpy as np
import wave
from ousia.session_processor import SessionProcessor


def load_wav_chunks(path: str, chunk_seconds: float = 2.0):
    """Yield (numpy_array, sample_rate) chunks from a WAV file."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0  # normalize to [-1, 1]
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)  # mono

    chunk_size = int(sr * chunk_seconds)
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size], sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic", action="store_true", help="Use live microphone")
    parser.add_argument("--file", default="samples/sample_speech.wav", help="WAV file for chunked streaming")
    args = parser.parse_args()

    sp = SessionProcessor(whisper_model="tiny")  # tiny = fast for CPU testing

    if args.mic:
        try:
            print("🎤 Live mic streaming (Ctrl+C to stop)...")
            for chunk, sr in SessionProcessor.mic_stream():
                sp.stream([(chunk, sr)])
        except KeyboardInterrupt:
            print("\nStopping...")
        except RuntimeError as e:
            print(f"Mic error: {e}")
            return
    else:
        print(f"📼 Streaming chunks from {args.file}...")
        chunks = list(load_wav_chunks(args.file, chunk_seconds=2.0))
        print(f"   ({len(chunks)} chunks @ ~2s each)")
        sp.stream(chunks)

    sp.graph.apply_session_decay()
    print("\n" + "=" * 50)
    print("GRAPH SUMMARY")
    print("=" * 50)
    sp.graph.summary()


if __name__ == "__main__":
    main()
