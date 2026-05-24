"""
Ousia Diarizer — Speaker Diarization via pyannote.audio.
Identifies who spoke when in an audio file.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class Diarizer:
    """
    Handles speaker diarization using pyannote.audio.
    Requires HF_TOKEN in environment for model access.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        hf_token: Hugging Face Read Token (needed to download pyannote models).
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.pipeline = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of the pyannote pipeline."""
        if self._initialized:
            return

        if not self.hf_token:
            print("  ⚠ No HF_TOKEN found. Diarization will be disabled (fallback to 'unknown' speaker).")
            self._initialized = True
            return

        try:
            from pyannote.audio import Pipeline
            import torch

            # Load the pretrained diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
            
            # Send to GPU if available, else CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.pipeline:
                self.pipeline.to(device)
            self._initialized = True
            print(f"  ✓ Diarizer initialized (device: {device})")
        except Exception as e:
            print(f"  ⚠ Failed to initialize pyannote pipeline: {e}")
            self._initialized = True

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Analyze audio and return segments with speaker labels.
        Returns: [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}, ...]
        """
        self._initialize()
        
        if not self.pipeline:
            return []

        try:
            output = self.pipeline(audio_path)
            segments = []
            
            # Check if it is a DiarizeOutput (common in some 3.x+ versions)
            # and potentially extract its internal annotation.
            annotation = output
            if not hasattr(annotation, "itertracks") and hasattr(output, "speaker_diarization"):
                annotation = output.speaker_diarization
            elif not hasattr(annotation, "itertracks") and hasattr(output, "annotation"):
                annotation = output.annotation
            
            # Handle standard pyannote Annotation object (or derived)
            if hasattr(annotation, "itertracks"):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            # Handle list or iterable of segments
            elif hasattr(output, "segments"):
                for seg in output.segments:
                    segments.append({
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "speaker": getattr(seg, "speaker", "unknown")
                    })
            elif isinstance(output, (list, tuple)):
                for seg in output:
                    if isinstance(seg, dict):
                        segments.append({
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "speaker": seg.get("speaker", "unknown")
                        })
                    else:
                        segments.append({
                            "start": getattr(seg, "start", 0),
                            "end": getattr(seg, "end", 0),
                            "speaker": getattr(seg, "speaker", "unknown")
                        })
            else:
                print(f"  ⚠ Unknown diarization output type: {type(output)}")
                # One last attempt: try to iterate directly
                try:
                    for turn, _, speaker in output:
                         segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                except:
                    pass
                
            return segments
        except Exception as e:
            print(f"  ⚠ Diarization error: {e}")
            return []

    @staticmethod
    def assign_speakers(utterances: List[Dict[str, Any]], diarization_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match Whisper utterances with diarization speakers based on highest overlap.
        """
        if not diarization_segments:
            for u in utterances:
                u["speaker"] = "unknown"
            return utterances

        for u in utterances:
            u_start, u_end = u["start"], u["end"]
            u_mid = (u_start + u_end) / 2
            
            # Find the speaker active at the midpoint of this utterance (simplest approach)
            best_speaker = "unknown"
            for d in diarization_segments:
                if d["start"] <= u_mid <= d["end"]:
                    best_speaker = d["speaker"]
                    break
            
            # Fallback: if midpoint doesn't hit, find closest segment
            if best_speaker == "unknown":
                min_dist = float('inf')
                for d in diarization_segments:
                    dist = min(abs(u_start - d["end"]), abs(u_end - d["start"]))
                    if dist < min_dist:
                        min_dist = dist
                        best_speaker = d["speaker"]
            
            u["speaker"] = best_speaker

        return utterances

if __name__ == "__main__":
    # Test stub
    d = Diarizer()
    print("Diarizer module loaded.")
