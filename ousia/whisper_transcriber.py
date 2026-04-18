"""
Whisper Transcriber — audio → transcript with word-level timestamps.

Prefers faster-whisper if installed (2-3× faster on CPU, same accuracy).
Falls back to openai-whisper otherwise. Both work on CPU.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def _wrap_faster_whisper(model) -> "WhisperTranscriber":
    """Return a wrapper exposing openai-whisper compatible .transcribe()."""
    class _FWWrapper:
        def __init__(self, fw_model):
            self._m = fw_model

        def transcribe(self, audio, word_timestamps: bool = False, verbose: bool = False) -> Dict[str, Any]:
            # faster-whisper accepts str path or numpy array
            segments, info = self._m.transcribe(
                audio,
                word_timestamps=word_timestamps,
                beam_size=1,  # faster; increase for accuracy if needed
            )
            seg_list = []
            for seg in segments:
                seg_dict = {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                }
                if word_timestamps and seg.words:
                    seg_dict["words"] = [
                        {"word": w.word, "start": w.start, "end": w.end} for w in seg.words
                    ]
                seg_list.append(seg_dict)
            return {
                "text": info.text if hasattr(info, "text") else " ".join(s["text"] for s in seg_list),
                "segments": seg_list,
                "language": info.language if hasattr(info, "language") else "en",
            }

    return _FWWrapper(model)


class WhisperTranscriber:
    def __init__(self, model_name: str = "base"):
        """
        model_name: tiny, base, small, medium, large
        - base: ~150MB, good balance for CPU
        - small: better accuracy, slower

        Uses faster-whisper if available (faster on CPU, same accuracy).
        Falls back to openai-whisper.
        """
        self.model_name = model_name
        self.model = None  # openai-whisper or wrapper

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            # int8 quantization works well on CPU; 'float16' needs GPU
            self.model = _wrap_faster_whisper(
                WhisperModel(model_name, device="cpu", compute_type="int8")
            )
            self._backend = "faster-whisper"
        except Exception:
            # Fall back to openai-whisper
            import whisper
            self.model = whisper.load_model(model_name)
            self._backend = "openai-whisper"

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Returns:
          {
            "text": str,           # full transcript
            "segments": [          # per-utterance chunks
              {
                "text": str,
                "start": float,    # seconds
                "end": float,
                "words": [...]     # if word_timestamps=True
              },
              ...
            ],
            "language": str
          }
        """
        # Both backends expose .transcribe() with same signature
        return self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )

    def get_utterances(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Convenience: return list of utterance dicts with text + timestamps.

        Each: {"text": str, "start": float, "end": float}
        """
        result = self.transcribe(audio_path)
        return [
            {
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
            }
            for seg in result.get("segments", [])
            if seg["text"].strip()
        ]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python whisper_transcriber.py <audio.wav>")
        sys.exit(1)

    t = WhisperTranscriber("base")
    utts = t.get_utterances(sys.argv[1])
    for u in utts:
        print(f"[{u['start']:.1f}s–{u['end']:.1f}s] {u['text']}")
