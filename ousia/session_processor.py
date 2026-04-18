"""
Session Processor — full pipeline wiring for one therapy session.

Audio → Whisper → Vocal Analysis → LLM → PatientGraph
"""

from typing import Optional, List, Dict, Any

from .whisper_transcriber import WhisperTranscriber
from .vocal_analyzer import VocalAnalyzer
from .llm_extractor import LLMExtractor
from .graph_engine import PatientGraph


class SessionProcessor:
    """
    Orchestrates the full Ousia pipeline for a single session.

    Usage:
        sp = SessionProcessor()
        graph = sp.process("session.wav")
        graph.summary()
    """

    def __init__(
        self,
        whisper_model: str = "base",
        baseline_utterances: int = 5,
        llm_model: str = "grok-3",
    ):
        """
        whisper_model: tiny/base/small/medium/large
        baseline_utterances: how many early utterances for vocal baseline
        llm_model: model name for OpenAI-compatible API
        """
        self.whisper = WhisperTranscriber(model_name=whisper_model)
        self.vocal = VocalAnalyzer(baseline_utterances=baseline_utterances)
        self.llm = LLMExtractor(model=llm_model)
        self.graph = PatientGraph()

        # Preload lazy resources (ConceptMerger embedding model) to avoid
        # latency spikes during first graph updates.
        if self.graph.preload():
            print("  ✓ ConceptMerger embedding model preloaded")
        else:
            print("  ℹ ConceptMerger: sentence-transformers not installed (merging disabled)")

    def process(
        self,
        audio_path: str,
        existing_graph: Optional[PatientGraph] = None,
    ) -> PatientGraph:
        """
        Run the full pipeline on an audio file.

        If existing_graph is provided, merges session into it (cumulative).
        Otherwise creates a fresh PatientGraph for this session.

        Returns: PatientGraph (updated cumulative graph)
        """
        if existing_graph is not None:
            self.graph = existing_graph

        # 1. Transcribe
        utterances: List[Dict[str, Any]] = self.whisper.get_utterances(audio_path)
        if not utterances:
            return self.graph

        # 2. Vocal analysis (baseline + per-utterance vocal_salience)
        vocal_results: List[Dict[str, Any]] = self.vocal.analyze(audio_path, utterances)

        # 3. Per-utterance: LLM extract → feed to graph
        prev_text: Optional[str] = None

        for i, utt in enumerate(utterances):
            text = utt["text"]
            # vocal_salience for this utterance (match by index; same order)
            try:
                vs = float(vocal_results[i].get("vocal_salience", 0.5))
            except Exception:
                vs = 0.5

            # LLM extraction (text-only: concepts, avoidance, valence, arousal)
            try:
                signals = self.llm.extract(text, context=prev_text)
            except Exception:
                signals = {"concepts": [], "avoidance": 0.0, "text_valence": 0.5, "text_arousal": 0.5}
            concepts: List[str] = signals.get("concepts", [])
            avoidance: float = float(signals.get("avoidance", 0.0))
            text_arousal: float = float(signals.get("text_arousal", 0.5))
            # Dissonance = |text arousal − voice arousal| (post-hoc merge, separation of concerns)
            dissonance: float = abs(text_arousal - vs)

            # 3a. Co-activation for all concept pairs in this utterance
            for a in range(len(concepts)):
                for b in range(a + 1, len(concepts)):
                    self.graph.record_coactivation(concepts[a], concepts[b], vs)

            # 3b. Avoidance → silence edges
            if avoidance >= self.graph.AVOIDANCE_THRESHOLD:
                for c in concepts:
                    self.graph.record_avoidance(c, avoidance)

            # 3c. Dissonance → stored on edges between concepts
            if dissonance > 0 and len(concepts) >= 2:
                for a in range(len(concepts)):
                    for b in range(a + 1, len(concepts)):
                        self.graph.record_dissonance(concepts[a], concepts[b], dissonance)

            prev_text = text

        # 4. End of session: apply use-dependent decay
        self.graph.apply_session_decay()

        return self.graph

    # ─── Streaming / Real-Time ─────────────────────────────────────────────────

    @staticmethod
    def _energy_vad(chunk: "np.ndarray", threshold: float = 0.01) -> bool:
        """
        Simple energy-based VAD (fallback if webrtcvad not installed).
        Returns True if chunk likely contains speech.
        """
        if chunk.size == 0:
            return False
        rms = float((chunk ** 2).mean()) ** 0.5
        return rms > threshold

    def stream(self, chunk_iter):
        """
        Streaming pipeline: process audio chunks incrementally.

        chunk_iter: iterator yielding (audio_np_array, sample_rate) tuples.
                    Each chunk is typically 1–3 seconds of audio.

        This method:
          - Transcribes each chunk (Whisper accepts numpy)
          - Extracts vocal features (openSMILE accepts numpy)
          - Runs LLM extraction
          - Updates graph incrementally (same as batch)
          - Handles failures per-chunk gracefully

        Returns: PatientGraph (cumulative, same as process())

        Usage (mic):
            for chunk, sr in SessionProcessor.mic_stream():
                sp.stream([ (chunk, sr) ])  # or collect and stream batches

        Usage (file chunks):
            # pre-split a long file into overlapping windows
            for chunk in windowed_audio:
                sp.stream([ (chunk, sr) ])
        """
        import numpy as np  # local import; already a dep
        import time

        prev_text = None  # for context carryover
        chunk_times = []
        early_feats_buffer: list[dict] = []  # buffer for baseline calibration
        baseline_calibrated = False

        for chunk, sr in chunk_iter:
            t0 = time.time()
            try:
                # 1. VAD: skip silent chunks
                if not self._energy_vad(chunk):
                    continue

                # 2. Whisper: transcribe this chunk (numpy array)
                t1 = time.time()
                try:
                    result = self.whisper.model.transcribe(
                        chunk.astype(np.float32),  # whisper expects float32
                        word_timestamps=False,
                        verbose=False,
                    )
                    text = (result.get("text") or "").strip()
                except Exception:
                    text = ""
                t_whisper = time.time() - t1

                if not text:
                    continue

                # 3. Vocal features: openSMILE on numpy
                t2 = time.time()
                try:
                    feats_df = self.vocal.smile.process_signal(chunk.astype(np.float32), sr)
                    row = feats_df.iloc[0]
                    feats = {col: float(row[col]) for col in row.index if col in self.vocal.KEY_FEATURES}

                    # Baseline calibration: buffer first N utterances, then compute mean
                    if not baseline_calibrated:
                        early_feats_buffer.append(feats)
                        if len(early_feats_buffer) >= self.vocal.baseline_utterances:
                            # Compute baseline from buffered features
                            baseline = {}
                            for f in self.vocal.KEY_FEATURES:
                                vals = [d.get(f, 0.0) for d in early_feats_buffer if f in d]
                                baseline[f] = float(np.mean(vals)) if vals else 0.0
                            self.vocal.baseline = baseline
                            baseline_calibrated = True
                        else:
                            # Not enough yet — use zeros baseline for now
                            if self.vocal.baseline is None:
                                self.vocal.baseline = {f: 0.0 for f in self.vocal.KEY_FEATURES}
                    vs = self.vocal.compute_salience(feats)
                except Exception:
                    vs = 0.5
                    feats = {}
                t_vocal = time.time() - t2

                # 4. LLM extraction (text-only: valence + arousal)
                t3 = time.time()
                try:
                    signals = self.llm.extract(text, context=prev_text)
                except Exception:
                    signals = {"concepts": [], "avoidance": 0.0, "text_valence": 0.5, "text_arousal": 0.5}
                t_llm = time.time() - t3

                concepts = signals.get("concepts", []) or []
                avoidance = float(signals.get("avoidance", 0.0))
                text_arousal = float(signals.get("text_arousal", 0.5))
                # Dissonance = |text arousal − voice arousal| (post-hoc, separation of concerns)
                dissonance = abs(text_arousal - vs)

                # 5. Update graph (same logic as batch)
                t4 = time.time()
                for a in range(len(concepts)):
                    for b in range(a + 1, len(concepts)):
                        self.graph.record_coactivation(concepts[a], concepts[b], vs)

                if avoidance >= self.graph.AVOIDANCE_THRESHOLD:
                    for c in concepts:
                        self.graph.record_avoidance(c, avoidance)

                if dissonance > 0 and len(concepts) >= 2:
                    for a in range(len(concepts)):
                        for b in range(a + 1, len(concepts)):
                            self.graph.record_dissonance(concepts[a], concepts[b], dissonance)
                t_graph = time.time() - t4

                prev_text = text
                t_total = time.time() - t0
                chunk_times.append(t_total)
                print(f"  chunk {len(chunk_times):2d}: whisper={t_whisper:.2f}s vocal={t_vocal:.2f}s llm={t_llm:.2f}s graph={t_graph:.3f}s total={t_total:.2f}s | '{text[:40]}...'")

            except Exception:
                continue

        if chunk_times:
            avg = sum(chunk_times) / len(chunk_times)
            print(f"\n  [timing] {len(chunk_times)} chunks, avg {avg:.2f}s/chunk, total {sum(chunk_times):.1f}s")

        # No decay mid-stream — caller should call apply_session_decay() at end
        return self.graph

    @staticmethod
    def mic_stream(sample_rate: int = 16000, chunk_seconds: float = 2.0):
        """
        Generator: yields (numpy_array, sample_rate) from default microphone.

        Uses sounddevice. If not installed, raises a clear error.

        Usage:
            sp = SessionProcessor()
            for chunk, sr in SessionProcessor.mic_stream():
                sp.stream([(chunk, sr)])
                # or accumulate and stream in batches
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "Streaming requires 'sounddevice'. Install with:\n"
                "  pip install sounddevice\n"
                "Then re-run."
            )

        blocksize = int(sample_rate * chunk_seconds)
        q = []

        def callback(indata, frames, time, status):
            if status:
                pass  # ignore overflow/underflow
            q.append(indata.copy())

        print("🎤 Streaming from microphone... (Ctrl+C to stop)")
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                            blocksize=blocksize, callback=callback):
            while True:
                if q:
                    chunk = q.pop(0)
                    yield chunk.flatten(), sample_rate

    # ─── Persistence helpers ────────────────────────────────────────────────────

    def save_graph(self, path: str) -> None:
        """Save current graph to JSON file (delegates to PatientGraph.save)."""
        self.graph.save(path)

    @classmethod
    def load_graph(cls, path: str) -> "PatientGraph":
        """Load a PatientGraph from JSON file."""
        return PatientGraph.load(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python session_processor.py <audio.wav>")
        print("Note: requires audio file with speech. For demo without audio, use graph_engine.py")
        sys.exit(1)

    sp = SessionProcessor()
    graph = sp.process(sys.argv[1])
    graph.summary()
