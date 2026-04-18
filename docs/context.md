# Ousia — Project Context

Ousia is a dynamic psychological graph engine designed for longitudinal analysis of therapy sessions. It builds an evolving network topology of a person's conceptual world based on their actual spoken words.

---

## The Core Idea

The system captures concepts mentioned by the patient and builds connections between them. These connections are not static; they evolve over time using neuro-inspired principles:
- **Hebbian Learning**: Connections strengthen when concepts co-occur, weighted by vocal intensity.
- **Use-Dependent Decay**: Frequently activated themes persist longer, while ephemeral mentions fade faster.
- **Long-Term Potentiation (LTP)**: Significantly strong connections become resistant to decay (modeling core beliefs or trauma).

---

## Key Differentiators

### Silence Nodes (Avoidance)
The engine detects avoidance signals (topic changes, vagueness, deflection) and creates edges to a special `silence` node. This reveals what the patient is *not* saying over time.

### Dissonance
By analyzing text and voice separately, the system measures "dissonance" (conflict). For example, if a patient says they are "fine" (positive text) but their voice shows high emotional arousal (tense voice), the system flags this conflict on the graph.

---

## Technical Stack

The project is built as a modular Python package:
- **Transcription**: `faster-whisper`
- **Vocal Analysis**: `openSMILE` (eGeMAPS features)
- **Conceptual Analysis**: OpenAI-compatible LLM API
- **Graph Engine**: Custom implementation using Python dataclasses and `NetworkX` for visualization
- **Persistence**: JSON snapshots (with a clear path to Neo4j)

---

## Implementation State

The current project represents a basic modular structure for the Ousia engine, including the core graph logic, a streaming session processor, and visualization tools.
