# Ousia

Ousia is a modular psychological graph engine. It builds an evolving network topology of a person's conceptual world based on their speech:

- **Nodes** = concepts, themes, people, emotions, and avoided topics.
- **Edges** = connections between concepts.
- **Weights** = influenced by co-occurrence and vocal salience.

The system is designed as a basic modular structure for therapy-related psychological analysis.

---

## Technical Overview

| Feature | Description |
|---------|-------------|
| **Silence Nodes** | Detects avoidance signals (topic changes, vagueness, etc.) and creates `silence ↔ concept` edges. |
| **Dissonance** | Measures the gap between text sentiment and vocal energy. High dissonance (e.g., "text says calm, voice says tense") is flagged for the therapist. |
| **Neuro-Inspired Dynamics** | Uses Hebbian learning, use-dependent decay, and LTP threshold to evolve the graph naturally. |
| **Sentence-Aware Chunking** | Processes audio in context-aware chunks to avoid splitting thoughts mid-sentence. |

---

## Neuro-Inspired Edge Dynamics

Instead of hardcoded formulas, Ousia uses three principles from synaptic plasticity:

### 1. Hebbian Learning (Edge Strength)
**"Neurons that fire together, wire together."**

```
Δw = LEARNING_RATE × vocal_salience
```

When two concepts co-activate in the same utterance chunk, the edge between them strengthens. `vocal_salience` = deviation from that person's baseline (pitch, tempo, energy). The signal itself decides the weight — no magic formula.

### 2. Use-Dependent Decay (Time Dynamics)
**Decay rate depends on activation frequency.**

```
decay_rate = BASE_DECAY / (1 + activation_count)
weight *= (1 - decay_rate)
```

High activation count → slow decay (persistent themes, trauma). Low activation → fast decay (ephemeral mentions). No hardcoded λ.

### 3. Long-Term Potentiation (LTP Threshold)
**If an edge passes a threshold, it becomes resistant to decay.**

```
if weight > LTP_THRESHOLD:
    decay_rate = 0
```

Strong edges "lock in." A deeply significant connection won't vanish just because it's unmentioned for a few sessions.

### Why This Works

| Problem | Old Approach | Neuro-Inspired |
|---------|--------------|----------------|
| Time decay | Hardcoded λ | Use-dependent (frequency decides) |
| Diminishing returns | `1/√n` or `log(1+n)` | Vocal deviation *is* the filter |
| Graph rewiring | Explicit rules | Emerges from Hebbian + decay + LTP |
| Hardcoded numbers | Many magic constants | Only 2: learning_rate, base_decay |

**Hyperparameters:** `LEARNING_RATE=0.1`, `BASE_DECAY=0.1`. These are starting points, not the rules.

---

## Silence & Dissonance (Differentiators)

### Silence Nodes (Avoidance Detection)

Avoidance signals trigger `silence ↔ concept` edges:

| Signal | Example | Detection |
|--------|---------|-----------|
| Topic change | "Anyway, how's the weather?" | Embedding similarity < threshold |
| Vagueness | "I don't know", "whatever" | LLM classification |
| Humor/deflection | Jokes after serious topic | LLM |
| Minimizing | "It's not a big deal" | LLM |
| Silence | Long pause after question | Audio: silence duration |

**Effect:** Repeated avoidance compounds (Hebbian). A `silence ↔ mother` edge with weight 0.8 means the patient strongly avoids mother.

**Visualization:** Dotted gray edges to `silence` node.

### Dissonance (Separation of Concerns)

Text and voice are analyzed independently, then merged:

```
dissonance = |text_arousal − vocal_salience|
```

| Signal | Source | Scale | Example |
|--------|--------|-------|---------|
| `text_valence` | LLM (text) | 0.0 (neg) ↔ 1.0 (pos) | "I'm devastated" = 0.1 |
| `text_arousal` | LLM (text) | 0.0 (calm) ↔ 1.0 (excited) | "Whatever" = 0.2, "I can't believe it!" = 0.9 |
| `vocal_salience` | openSMILE (audio) | 0.0 (baseline) ↔ 1.0 (extreme deviation) | High jitter/energy = 0.8 |

**Why arousal?** Not just "calm ↔ tense" — arousal captures energy level. Shaky voice, flat monotone, breathy, animated — all are arousal deviations. Valence (positive↔negative) is future-proofed via `text_valence` but not yet used in dissonance (voice valence harder to extract from eGeMAPS).

**Effect:** Edge attribute `dissonance ∈ [0,1]`. In visualization: red gradient or tooltip.

**Thresholds:** Dissonance > 0.6 highlighted. Avoidance > 0.5 creates silence edge.

---

## Single LLM Call Per Chunk (Text-Only)

One LLM call per utterance receives **text only** and returns:

```json
{
  "concepts": ["mother", "childhood", "home"],
  "avoidance": 0.8,
  "avoidance_types": ["vagueness", "topic_abort"],
  "text_valence": 0.2,
  "text_arousal": 0.7
}
```

Voice is analyzed separately by `VocalAnalyzer` (openSMILE eGeMAPS → `vocal_salience`). Dissonance is computed post-hoc in `SessionProcessor`:

```
dissonance = |text_arousal − vocal_salience|
```

Then:
- Concepts → `record_coactivation()` for each pair
- Avoidance > 0.5 → `record_avoidance("silence", concept, avoidance)`
- Dissonance > 0 → `record_dissonance(concept_a, concept_b, dissonance)`

---

## Context Between Chunks (Sentence Breaks)

**Problem:** Whisper chunks by time. Sentence "I was thinking about my mother and how she used to— wait, actually never mind." splits mid-thought.

**Solution:** Sentence-aware chunking + light context carryover

1. Chunk at sentence boundaries (Whisper punctuation + NLTK)
2. For each utterance, pass previous 1 sentence as context in prompt
3. If utterance < 3 words and no punctuation, merge with next

Result: LLM sees "wait, actually never mind" with context → detects avoidance (topic aborted).

---

## Architecture: Development Phases

```
PHASE 1 — Graph Engine (NEURO-INSPIRED CORE) ✅ DONE
────────────────────────────────────────────────────
• Hebbian learning, use-dependent decay, LTP threshold
• Silence nodes: record_avoidance("silence", concept, score)
• Dissonance: Edge.dissonance attribute (max observed)
• Multi-chunk avoidance compounds (Hebbian)
• Prototype: graph_engine.py (demo runs, proves rules work)
• Hyperparams: LEARNING_RATE, BASE_DECAY
• Thresholds: DISSONANCE_THRESHOLD=0.6, AVOIDANCE_THRESHOLD=0.5

Next: importable module for the pipeline.


PHASE 2 — Concept Extraction Pipeline
─────────────────────────────────────
Input: raw transcript (or audio → Whisper)
Output: list of concepts (patient's own words)

Sub-steps:
  2a. Whisper transcription (with word-level timestamps)
      - Input: .wav audio
      - Output: [{"text": "...", "start": 0.0, "end": 3.2}, ...]

  2b. Per-utterance concept extraction (LLM, structured JSON)
      - Prompt: "What is this person talking about? Use their exact words."
      - Output: ["work", "boss", "angry", "stressed"]

  2c. Node merging (embeddings) ✅ DONE
      - Problem: "I'm overwhelmed" vs "I feel overwhelmed"
      - Solution: ConceptMerger in graph_engine.py
        - Tier 1: sentence-transformers cosine ≥ 0.88
        - Tier 2: Jaccard token overlap > 0 (guards against "angry at boss" ≈ "angry at mother")
      - Merge strategy: first utterance wins (canonical form)
      - Graceful fallback: if sentence-transformers not installed, no-op


PHASE 3 — Vocal Analysis & Baseline
───────────────────────────────────
Input: audio + transcript (with timestamps)
Output: per-utterance vocal_salience (0.0–1.0)

Sub-steps:
  3a. Per-person baseline calibration
      - First N utterances (or first session) → mean pitch, tempo, jitter, shimmer, energy, pauses
      - Store baseline in patient profile

  3b. Feature extraction (openSMILE eGeMAPS)
      - Per utterance: pitch (f0), tempo, jitter, shimmer, energy variance, pause ratio
      - Compare to baseline → deviation percentages

  3c. Vocal salience scoring
      - Combine deviations into single scalar [0,1]
      - Output: vocal_salience for each utterance (fed to Hebbian)


PHASE 4 — Session Processor (Pipeline Integration) ✅ DONE
──────────────────────────────────────────────────
Input: audio file (one therapy session) **or** live audio chunks (streaming)
Output: updated cumulative PatientGraph

**Batch (file):**
  `SessionProcessor.process(audio_path)` — processes whole file at once

**Streaming (real-time):**
  `SessionProcessor.stream(chunk_iter)` — processes chunks incrementally
  `SessionProcessor.mic_stream()` — yields mic audio chunks (requires `sounddevice`)
  → Same graph logic, same `record_*` calls, just fed live

Flow (both paths):
  Audio → Whisper → utterances (text + timestamps)
       ↓
  For each utterance:
    ├─ Extract concepts (LLM, 1 call) → list of strings
    ├─ Extract vocal features (openSMILE eGeMAPS) → vocal_salience
    ├─ LLM returns: concepts, text_valence, text_arousal, avoidance (text-only)
    ├─ Dissonance = |text_arousal − vocal_salience| (post-hoc, no LLM voice input)
    ├─ For each pair: graph.record_coactivation(a, b, vocal_salience)
    ├─ If avoidance > 0.5: graph.record_avoidance("silence", concept, avoidance)
    └─ If dissonance > 0: graph.record_dissonance(a, b, dissonance)
       ↓
  At end of session:
    └─ graph.apply_session_decay()
       ↓
  Return: updated PatientGraph (ready to persist)


PHASE 5 — Storage & Persistence ✅ DONE
─────────────────────────────────────────
Store per-patient state across sessions.

Entities:
  • Patient: id, name, baseline (pitch/tempo/...), graph (JSON or pickled)
  • Session: id, patient_id, audio_path, transcript, timestamp
  • GraphSnapshot: patient_id, session_id, serialized graph

Storage strategy:
  • MVP: JSON-first (PatientGraph.save/load). Zero infra, clean path to Neo4j.
  • Production: Neo4j (native graph model; Cypher queries; multi-patient; graph algorithms).
  • Alternative: PostgreSQL + JSONB or SQLite (simple but not graph-native).

**Implemented:**
- `PatientGraph.to_dict()` / `from_dict()` — serialize nodes, edges, hyperparams
- `PatientGraph.save(path)` / `load(path)` — JSON file I/O
- `SessionProcessor.save_graph(path)` / `load_graph(path)` — convenience wrappers
- ConceptMerger excluded (runtime embedding model, recreated on load)

Key: each session appends to cumulative graph. Snapshot list enables prev/next timeline demo.


PHASE 6 — Graph Visualization
─────────────────────────────
Therapist-facing interactive graph.

Snapshot timeline (demo):
  • Capture graph state after each utterance/chunk via `copy.deepcopy(graph)`
  • Prev/next navigation shows graph growth, rewiring, decay over time
  • Zero infra — pure Python list of graph states

Backend:
  • Serialize graph to JSON: nodes [{id, label, weight}], edges [{source, target, weight, dissonance}]

Frontend:
  • D3.js or vis.js — interactive zoom, pan, hover
  • NetworkX (Python) — static images for quick view

Visual encoding:
  • Node size = activation count
  • Edge thickness = weight
  • Edge style: solid = normal, dotted = silence
  • Edge color: red gradient = dissonance
  • Node color: avoidance = gray tint


PHASE 7 — Backend API (FastAPI)
───────────────────────────────
HTTP interface for the full pipeline.

Endpoints (MVP):
  POST /patients              → create patient
  POST /patients/{id}/sessions → upload audio → run pipeline → return graph
  GET  /patients/{id}/graph   → return current cumulative graph (JSON)
  GET  /patients/{id}/history → list past sessions

Tech: FastAPI + uvicorn


PHASE 8 — Frontend (Optional for MVP)
─────────────────────────────────────
Polished therapist UI.

Options:
  • React + D3.js (graph visualization)
  • Vanilla JS + vis.js (lighter)
  • Streamlit (Python, rapid proto)

For 20-hour MVP: skip or minimal. Backend + simple viz is enough.


PHASE 9 — Testing, Edge Cases, Polish
─────────────────────────────────────
• Multi-session simulation (graph accumulates correctly?)
• Edge case: patient says nothing (empty session)
• Edge case: very short utterances (no co-activations)
• Baseline: what if first session is atypical?
• Performance: Whisper + LLM latency
• Therapist feedback loop (does the graph make sense?)


┌─────────────────────────────────────────────────────────────────────────────┐
│                              MVP SCOPE (20 HOURS)                            │
└─────────────────────────────────────────────────────────────────────────────┘

Minimum viable for demo:
  ✅ Phase 1 — Graph Engine (Hebbian, use-dependent decay, LTP, silence nodes, dissonance)
  ✅ Phase 2a — WhisperTranscriber (faster-whisper primary, openai-whisper fallback)
  ✅ Phase 2b — LLMExtractor (single JSON call: concepts + text_valence + text_arousal + avoidance; dissonance computed post-hoc)
  ✅ Phase 2c — ConceptMerger (cosine + Jaccard, preloaded in SessionProcessor)
  ✅ Phase 3 — VocalAnalyzer (openSMILE eGeMAPSv02, per-person baseline calibration)
  ✅ Phase 4 — SessionProcessor (batch `process(file)`, streaming `stream(chunks)`, `mic_stream()`)
  ✅ Streaming pipeline (VAD, per-chunk timing, graceful fallback on errors)
  ✅ LLM graceful fallback (neutral signals if no API key; works offline)
  ✅ ConceptMerger preload (embedding model loaded at init; no first-call latency spike)
  ✅ Phase 5 — Persistence (JSON-first: PatientGraph.save/load, SessionProcessor.save_graph/load_graph)
  🔲 Phase 6 — Visualization (snapshot prev/next demo is natural first step)
  🔲 Phase 7 — Backend API (FastAPI)

→ MVP = "batch file or live mic → Whisper → openSMILE → LLM → ConceptMerger → PatientGraph"
   (session_processor.py: process(file) or stream(chunks))

The neuro-inspired engine (Phase 1) + silence/dissonance detection (Phase 2b) is the core. The rest is wiring.
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core Engine | Python (pure stdlib + dataclasses) |
| Transcription | faster-whisper (primary), openai-whisper (fallback) |
| Concept/Avoidance/Dissonance | LLM (single JSON call per utterance; OpenAI compatible) |
| Vocal Features | openSMILE eGeMAPSv02 (88 prosodic features, per-person baseline) |
| Embeddings (node merge) | sentence-transformers (all-MiniLM-L6-v2, preloaded) |
| Graph | Pure Python dataclasses (`PatientGraph`); NetworkX optional for viz |
| Storage | ✅ JSON snapshots (`PatientGraph.save/load`); Neo4j (production, Phase 7) |
| API | FastAPI + uvicorn (Phase 7) |
| Frontend | React / vis.js / D3.js (Phase 8; optional for MVP) |

---

## How to Run (Current Prototype)

**Quick demo (graph engine only):**
```bash
python graph_engine.py
```
Shows Hebbian learning, avoidance, dissonance, and decay with hardcoded values.

**Full pipeline (real audio → graph):**
```bash
# Batch: process a WAV file
python session_processor.py sample_speech.wav

# Streaming test (chunks a sample WAV, simulates real-time)
python test_realtime.py

# Live mic streaming (requires sounddevice)
pip install sounddevice
python test_realtime.py --mic
```

Both `process(file)` and `stream(chunks)` wire: Whisper → openSMILE → LLM → PatientGraph (with ConceptMerger, silence nodes, dissonance). No API key required for basic run (LLM falls back gracefully).

**Persist the graph (save across sessions):**
```python
from graph_engine import PatientGraph
from session_processor import SessionProcessor

# After processing, save the cumulative graph
sp = SessionProcessor()
graph = sp.process("session.wav")
graph.save("patient.json")  # or: sp.save_graph("patient.json")

# Later: load and continue
graph = PatientGraph.load("patient.json")
graph.record_coactivation("new", "concept", 0.7)
graph.save("patient.json")
```

---

## Project Timeline

- **Version:** 0.1.0
- **Status:** Basic modular structure setup.

---

*This README documents the full Ousia architecture and design decisions from the project context and development sessions.*
