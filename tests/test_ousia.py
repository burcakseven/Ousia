"""
Ousia Comprehensive Test Suite

Run: python test_ousia.py

Covers:
  1. Graph Engine (ConceptMerger, Edge, PatientGraph)
  2. Vocal Analyzer (features, baseline, salience)
  3. Whisper Transcriber (transcribe, get_utterances)
  4. Session Processor (process, stream)
  5. Integration (end-to-end with sample audio)
"""

import sys
import traceback


# ─── Helpers ───────────────────────────────────────────────────────────────────

def ok(name: str):
    print(f"  ✓ {name}")

def fail(name: str, err: str):
    print(f"  ✗ {name}")
    print(f"    Error: {err}")

def section(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─── Test 1: Graph Engine ──────────────────────────────────────────────────────

def test_graph_engine():
    section("1. Graph Engine Tests")

    from ousia.graph_engine import ConceptMerger, Edge, PatientGraph

    # ── ConceptMerger ──
    try:
        cm = ConceptMerger(cosine_threshold=0.88, min_jaccard=0.01)
        # Without sentence-transformers, merge is no-op
        result = cm.merge("I'm overwhelmed", ["I feel overwhelmed", "work stress"])
        ok("ConceptMerger.merge() (no-op fallback if no sentence-transformers)")
    except Exception as e:
        fail("ConceptMerger.merge()", str(e))

    # ── Edge ──
    try:
        e = Edge()
        e.hebbian_update(vocal_salience=0.8, learning_rate=0.1)
        assert 0 < e.weight <= 1.0, f"weight out of range: {e.weight}"
        assert e.activation_count == 1
        ok("Edge.hebbian_update()")
    except Exception as e:
        fail("Edge.hebbian_update()", str(e))

    try:
        e = Edge(weight=0.9, activation_count=10)
        e.apply_decay(base_decay=0.1, ltp_threshold=0.8)
        # weight > LTP_THRESHOLD → decay_rate=0 → weight unchanged
        assert e.weight == 0.9, f"LTP should lock weight, got {e.weight}"
        ok("Edge.apply_decay() with LTP lock")
    except Exception as e:
        fail("Edge.apply_decay() (LTP)", str(e))

    try:
        e = Edge(weight=0.3, activation_count=1)
        e.apply_decay(base_decay=0.1, ltp_threshold=0.8)
        # weight < LTP → decay applies
        assert e.weight < 0.3, f"decay should reduce weight, got {e.weight}"
        ok("Edge.apply_decay() with decay")
    except Exception as e:
        fail("Edge.apply_decay() (decay)", str(e))

    try:
        e = Edge()
        e.record_dissonance(0.7)
        e.record_dissonance(0.3)
        assert e.dissonance == 0.7, f"should keep max, got {e.dissonance}"
        ok("Edge.record_dissonance() (max tracking)")
    except Exception as e:
        fail("Edge.record_dissonance()", str(e))

    # ── PatientGraph ──
    try:
        g = PatientGraph()
        g.record_coactivation("work", "boss", 0.9)
        g.record_coactivation("work", "stressed", 0.8)
        assert "work" in g.nodes
        assert "boss" in g.nodes
        assert len(g.edges) >= 1
        ok("PatientGraph.record_coactivation()")
    except Exception as e:
        fail("PatientGraph.record_coactivation()", str(e))

    try:
        g = PatientGraph()
        g.record_dissonance("work", "fine", 0.85)
        assert len(g.edges) >= 1
        ok("PatientGraph.record_dissonance()")
    except Exception as e:
        fail("PatientGraph.record_dissonance()", str(e))

    try:
        g = PatientGraph()
        g.record_avoidance("mother", 0.7)
        assert "silence" in g.nodes
        assert "mother" in g.nodes
        ok("PatientGraph.record_avoidance() → silence edge")
    except Exception as e:
        fail("PatientGraph.record_avoidance()", str(e))

    try:
        g = PatientGraph()
        for _ in range(3):
            g.record_coactivation("work", "stressed", 0.6)
        g.apply_session_decay()
        ok("PatientGraph.apply_session_decay()")
    except Exception as e:
        fail("PatientGraph.apply_session_decay()", str(e))

    try:
        g = PatientGraph()
        g.record_coactivation("work", "boss", 0.9)
        g.record_coactivation("work", "stressed", 0.8)
        g.record_avoidance("mother", 0.7)
        g.summary()  # should print without error
        ok("PatientGraph.summary()")
    except Exception as e:
        fail("PatientGraph.summary()", str(e))


# ─── Test 2: Vocal Analyzer ────────────────────────────────────────────────────

def test_vocal_analyzer():
    section("2. Vocal Analyzer Tests")

    from ousia.vocal_analyzer import VocalAnalyzer
    import os

    # ── Init ──
    try:
        va = VocalAnalyzer(baseline_utterances=5)
        assert va.smile is not None
        assert len(va.KEY_FEATURES) == 8
        ok("VocalAnalyzer.__init__()")
    except Exception as e:
        fail("VocalAnalyzer.__init__()", str(e))

    # ── compute_salience without baseline ──
    try:
        va = VocalAnalyzer()
        sal = va.compute_salience({"F0semitoneFrom27.5Hz_sma3nz_amean": 10.0})
        assert sal == 0.5, f"no baseline → neutral 0.5, got {sal}"
        ok("VocalAnalyzer.compute_salience() (no baseline → 0.5)")
    except Exception as e:
        fail("VocalAnalyzer.compute_salience() (no baseline)", str(e))

    # ── analyze() with sample audio ──
    audio_path = "samples/sample_speech.wav"
    if os.path.exists(audio_path):
        try:
            va = VocalAnalyzer(baseline_utterances=2)
            # Need utterances with start/end — fake them
            fake_utts = [
                {"text": "hello world", "start": 0.0, "end": 2.0},
                {"text": "testing vocal", "start": 2.0, "end": 4.0},
                {"text": "more speech", "start": 4.0, "end": 6.0},
            ]
            results = va.analyze(audio_path, fake_utts)
            assert len(results) == 3
            for r in results:
                assert "vocal_salience" in r
                assert 0.0 <= r["vocal_salience"] <= 1.0
            ok("VocalAnalyzer.analyze() (with sample audio)")
        except Exception as e:
            fail("VocalAnalyzer.analyze() (sample audio)", str(e))
    else:
        print("  ℹ sample_speech.wav not found — skipping audio-dependent tests")


# ─── Test 3: Whisper Transcriber ───────────────────────────────────────────────

def test_whisper_transcriber():
    section("3. Whisper Transcriber Tests")

    from ousia.whisper_transcriber import WhisperTranscriber
    import os

    # ── Init (faster-whisper or openai-whisper) ──
    try:
        wt = WhisperTranscriber(model_name="tiny")
        assert wt.model is not None
        ok(f"WhisperTranscriber.__init__() (backend={wt._backend})")
    except Exception as e:
        fail("WhisperTranscriber.__init__()", str(e))

    # ── get_utterances with sample audio ──
    audio_path = "samples/sample_speech.wav"
    if os.path.exists(audio_path):
        try:
            wt = WhisperTranscriber(model_name="tiny")
            utts = wt.get_utterances(audio_path)
            assert isinstance(utts, list)
            for u in utts:
                assert "text" in u and "start" in u and "end" in u
            ok(f"WhisperTranscriber.get_utterances() ({len(utts)} utterances)")
        except Exception as e:
            fail("WhisperTranscriber.get_utterances() (sample audio)", str(e))
    else:
        print("  ℹ sample_speech.wav not found — skipping audio-dependent tests")


# ─── Test 4: Session Processor (graceful without API key) ──────────────────────

def test_session_processor_no_api():
    section("4. Session Processor Tests (graceful without API key)")

    from ousia.session_processor import SessionProcessor

    # ── Init succeeds without API key (graceful fallback) ──
    try:
        sp = SessionProcessor(whisper_model="tiny")
        ok("SessionProcessor.__init__() succeeds (graceful LLM fallback)")
    except Exception as e:
        fail("SessionProcessor.__init__()", str(e))
        return

    # ── LLMExtractor.extract() returns neutral signals without API key ──
    try:
        from ousia.llm_extractor import LLMExtractor
        llm = LLMExtractor()
        result = llm.extract("test utterance")
        assert result["concepts"] == []
        assert result["text_valence"] == 0.5
        assert result["text_arousal"] == 0.5
        ok("LLMExtractor.extract() returns neutral signals (no API key)")
    except Exception as e:
        fail("LLMExtractor.extract() (no API key)", str(e))


# ─── Test 5: Integration (test_realtime.py style) ──────────────────────────────

def test_integration():
    section("5. Integration Tests (requires API key)")

    import os
    has_key = os.getenv("XAI_API_KEY")

    if not has_key:
        print("  ℹ No XAI_API_KEY set — skipping full integration test")
        print("    (Set XAI_API_KEY to test end-to-end)")
        return

    try:
        from tests.test_realtime import load_wav_chunks
        from ousia.session_processor import SessionProcessor

        chunks = list(load_wav_chunks("sample_speech.wav", chunk_seconds=2.0))
        assert len(chunks) > 0
        ok(f"load_wav_chunks() ({len(chunks)} chunks)")

        sp = SessionProcessor(whisper_model="tiny")
        sp.stream(chunks[:3])  # test first 3 chunks only (faster)
        sp.graph.apply_session_decay()
        sp.graph.summary()
        ok("SessionProcessor.stream() + apply_session_decay() + summary()")
    except Exception as e:
        fail("Integration test", str(e))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("OUSIA TEST SUITE")
    print("="*60)

    test_graph_engine()
    test_vocal_analyzer()
    test_whisper_transcriber()
    test_session_processor_no_api()
    test_integration()

    print("\n" + "="*60)
    print("DONE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
