"""
Vocal Analyzer — audio + timestamps → per-utterance vocal_salience (0.0–1.0).

Uses openSMILE eGeMAPS features (88 prosodic/voice-quality features).
Compares each utterance to per-person baseline → deviation score.

No librosa. openSMILE is the better choice for this use case.
"""

import opensmile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class VocalAnalyzer:
    """
    Extracts vocal features via openSMILE and computes deviation from baseline.

    vocal_salience = how much this utterance deviates from the person's normal voice.
    High = tense/stressed/animated. Low = flat/calm.
    """

    # Key eGeMAPS features for vocal affect (subset, not all 88)
    KEY_FEATURES = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",   # pitch mean
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",  # pitch variability
        "loudness_sma3_amean",                  # energy mean
        "loudness_sma3_stddevNorm",             # energy variability
        "jitterLocal_sma3nz_amean",             # jitter
        "shimmerLocaldB_sma3nz_amean",          # shimmer
        "HNRdBACF_sma3nz_amean",                # harmonics-to-noise
        "alphaRatioV_sma3nz_amean",             # spectral balance
    ]

    def __init__(self, baseline_utterances: int = 5):
        """
        baseline_utterances: how many early utterances to use for baseline calibration.
        """
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.baseline_utterances = baseline_utterances
        self.baseline: Optional[Dict[str, float]] = None

    def extract_features(self, audio_path: str, start: float, end: float) -> Dict[str, float]:
        """
        Extract eGeMAPS functionals for a time segment [start, end] seconds.
        Returns dict of feature_name → value.
        """
        # openSMILE can process a segment via start/end
        feats = self.smile.process_file(
            audio_path,
            start=start,
            end=end,
        )
        # feats is a DataFrame with one row
        row = feats.iloc[0]
        return {col: float(row[col]) for col in row.index if col in self.KEY_FEATURES}

    def calibrate_baseline(self, audio_path: str, utterances: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Build baseline from first N utterances.
        Returns mean feature values.
        """
        feats_list = []
        for u in utterances[: self.baseline_utterances]:
            try:
                feats = self.extract_features(audio_path, u["start"], u["end"])
                feats_list.append(feats)
            except Exception:
                continue

        if not feats_list:
            # fallback: neutral baseline
            return {f: 0.0 for f in self.KEY_FEATURES}

        # Average across utterances
        baseline = {}
        for f in self.KEY_FEATURES:
            vals = [d.get(f, 0.0) for d in feats_list if f in d]
            baseline[f] = float(np.mean(vals)) if vals else 0.0
        self.baseline = baseline
        return baseline

    def compute_salience(self, feats: Dict[str, float]) -> float:
        """
        Compute vocal_salience (0.0–1.0) from features vs baseline.

        Simple approach: normalize deviation per feature, average.
        High = deviates a lot (tense, stressed, animated).
        Low = close to baseline (calm, flat).
        """
        if self.baseline is None:
            # no baseline yet; treat as neutral
            return 0.5

        deviations = []
        for f in self.KEY_FEATURES:
            if f not in feats or f not in self.baseline:
                continue
            base = self.baseline[f]
            val = feats[f]
            # relative deviation (avoid div0)
            if abs(base) < 1e-6:
                dev = abs(val)
            else:
                dev = abs(val - base) / (abs(base) + 1e-6)
            # clamp per-feature deviation to [0, 2] then normalize to [0,1]
            dev = min(dev, 2.0) / 2.0
            deviations.append(dev)

        if not deviations:
            return 0.5

        # Average deviation → vocal_salience
        return float(np.clip(np.mean(deviations), 0.0, 1.0))

    def analyze(self, audio_path: str, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Full pipeline: calibrate baseline, then for each utterance compute vocal_salience.

        Returns list of dicts:
          [{"text": ..., "start": ..., "end": ..., "vocal_salience": 0.0–1.0, "features": {...}}, ...]
        """
        # Calibrate on first utterances
        self.calibrate_baseline(audio_path, utterances)

        results = []
        for u in utterances:
            try:
                feats = self.extract_features(audio_path, u["start"], u["end"])
                salience = self.compute_salience(feats)
            except Exception:
                feats = {}
                salience = 0.5  # neutral fallback

            results.append({
                "text": u["text"],
                "start": u["start"],
                "end": u["end"],
                "vocal_salience": round(salience, 3),
                "features": feats,
            })

        return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vocal_analyzer.py <audio.wav>")
        sys.exit(1)

    # Need utterances too — just demo with fake for now
    print("VocalAnalyzer requires utterances (use with session_processor).")
    print("Direct test: requires audio + pre-computed utterance timestamps.")
