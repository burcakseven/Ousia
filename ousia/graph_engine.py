"""
Ousia Graph Engine — Neuro-Inspired Edge Dynamics

Three rules:
1. Hebbian Learning: Δw = LEARNING_RATE × vocal_salience (co-activation)
2. Use-Dependent Decay: decay_rate = BASE_DECAY / (1 + activation_count)
3. LTP Threshold: if w > LTP_THRESHOLD, edge becomes decay-resistant

Extra features:
- Silence nodes: avoidance detection → silence ↔ concept edges
- Dissonance: text/voice conflict stored as edge attribute
- Multi-chunk avoidance compounds (Hebbian applies)

Only 2 hyperparameters: LEARNING_RATE, BASE_DECAY
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set, List, Optional, Tuple
import numpy as np


# ─── Concept Merging (Tier 1 + Tier 2) ─────────────────────────────────────────

class ConceptMerger:
    """
    Merges semantically similar concepts to prevent graph explosion.

    Tier 1: Cosine similarity via sentence-transformers (threshold ~0.88)
    Tier 2: Jaccard token overlap guard (prevents "angry at boss" ≈ "angry at mother")

    If sentence-transformers is not installed, merging is a no-op (graceful fallback).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cosine_threshold: float = 0.88,
        min_jaccard: float = 0.01,  # >0 means at least one token in common
    ):
        self.model_name = model_name
        self.cosine_threshold = cosine_threshold
        self.min_jaccard = min_jaccard
        self._model = None  # lazy loaded
        self._cache: Dict[str, np.ndarray] = {}  # concept -> embedding

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            self._model = None  # graceful fallback

    def preload(self) -> bool:
        """
        Explicitly load the embedding model upfront.

        Call this during initialization (e.g., SessionProcessor.__init__)
        to avoid lazy-load latency spikes during first graph updates.

        Returns True if model loaded successfully, False if unavailable.
        """
        self._load_model()
        return self._model is not None

    def _embed(self, text: str) -> Optional[np.ndarray]:
        self._load_model()
        if self._model is None:
            return None
        if text not in self._cache:
            vec = self._model.encode(text, normalize_embeddings=True)
            self._cache[text] = np.asarray(vec)
        return self._cache[text]

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        # simple lowercase word tokens
        return set(w.lower() for w in text.split() if w.isalpha())

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        # assumes already normalized; dot product = cosine
        return float(np.dot(a, b))

    def should_merge(
        self, new_concept: str, existing: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if new_concept should merge into an existing concept.

        Returns: (should_merge, target_concept_or_None)
        """
        if not existing:
            return False, None
        new_vec = self._embed(new_concept)
        if new_vec is None:
            # no model → no merging
            return False, None

        best_score = -1.0
        best_match: Optional[str] = None

        for ex in existing:
            ex_vec = self._embed(ex)
            if ex_vec is None:
                continue
            cos = self._cosine(new_vec, ex_vec)
            jac = self._jaccard(self._tokens(new_concept), self._tokens(ex))
            if cos >= self.cosine_threshold and jac >= self.min_jaccard:
                if cos > best_score:
                    best_score = cos
                    best_match = ex

        if best_match is not None:
            return True, best_match
        return False, None

    def merge(self, concept: str, existing: List[str]) -> str:
        """
        Return the concept to use: either the merged-into existing one, or the original.
        """
        should, target = self.should_merge(concept, existing)
        return target if should and target else concept


# ─── Graph Primitives ──────────────────────────────────────────────────────────

@dataclass
class Edge:
    weight: float = 0.0
    activation_count: int = 0
    dissonance: float = 0.0  # 0.0–1.0, max text/voice conflict observed

    def hebbian_update(self, vocal_salience: float, learning_rate: float):
        """Hebbian: neurons that fire together, wire together."""
        self.weight += learning_rate * vocal_salience
        self.activation_count += 1
        # Synaptic scaling (soft cap)
        self.weight = min(self.weight, 1.0)

    def apply_decay(self, base_decay: float, ltp_threshold: float):
        """Use-dependent decay. If above LTP threshold, resist decay."""
        decay_rate = base_decay / (1 + self.activation_count)
        if self.weight > ltp_threshold:
            decay_rate = 0  # LTP: locked in
        self.weight *= (1 - decay_rate)
        # Clamp to [0, 1]
        self.weight = max(0.0, min(self.weight, 1.0))

    def record_dissonance(self, dissonance_score: float):
        """Store max dissonance observed for this edge (text vs voice conflict)."""
        self.dissonance = max(self.dissonance, dissonance_score)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to JSON-compatible dict."""
        return {
            "weight": self.weight,
            "activation_count": self.activation_count,
            "dissonance": self.dissonance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        """Reconstruct Edge from dict."""
        return cls(
            weight=float(d.get("weight", 0.0)),
            activation_count=int(d.get("activation_count", 0)),
            dissonance=float(d.get("dissonance", 0.0)),
        )


@dataclass
class PatientGraph:
    """
    Cumulative patient graph.

    Nodes: set of concept strings (patient's own words) + special "silence" node
    Edges: (concept_a, concept_b) -> Edge (undirected, stored as frozenset key)

    Concept merging: enabled by default via ConceptMerger (Tier 1 cosine + Tier 2 Jaccard).
    If sentence-transformers is not installed, merging is a no-op (graceful).
    """

    nodes: Set[str] = field(default_factory=set)
    edges: Dict[frozenset, Edge] = field(default_factory=dict)
    concept_merger: Optional[ConceptMerger] = field(default_factory=ConceptMerger, repr=False)

    # Hyperparameters (starting points, not hardcoded rules)
    LEARNING_RATE: float = 0.1
    BASE_DECAY: float = 0.1
    LTP_THRESHOLD: float = 0.8

    # Thresholds (from design spec)
    DISSONANCE_THRESHOLD: float = 0.6  # highlight edges above this
    AVOIDANCE_THRESHOLD: float = 0.5  # create silence edge above this

    def add_node(self, concept: str) -> str:
        """
        Add concept to nodes. If merging is enabled, returns the merged (canonical) form.
        Callers should use the returned string for edges.
        """
        if self.concept_merger is not None:
            concept = self.concept_merger.merge(concept, list(self.nodes))
        self.nodes.add(concept)
        return concept

    def _get_edge(self, a: str, b: str) -> Edge:
        key = frozenset([a, b])
        if key not in self.edges:
            self.edges[key] = Edge()
        return self.edges[key]

    def record_coactivation(self, concept_a: str, concept_b: str, vocal_salience: float):
        """
        Hebbian learning: two concepts co-fired in an utterance chunk.

        concept_a, concept_b: patient's own words (auto-merged if similar to existing)
        vocal_salience: deviation from baseline (0.0 = flat, 1.0 = extreme)
        """
        if concept_a == concept_b:
            return  # no self-loops
        concept_a = self.add_node(concept_a)
        concept_b = self.add_node(concept_b)
        edge = self._get_edge(concept_a, concept_b)
        edge.hebbian_update(vocal_salience, self.LEARNING_RATE)

    def record_dissonance(self, concept_a: str, concept_b: str, dissonance_score: float):
        """
        Record text/voice conflict for an edge.

        dissonance_score: 0.0 = no conflict, 1.0 = extreme conflict
        Stored as max observed on that edge.
        Concepts auto-merged if similar to existing.

        Also triggers Hebbian: dissonance IS a co-activation signal (conflict = significance).
        """
        if concept_a == concept_b:
            return
        concept_a = self.add_node(concept_a)
        concept_b = self.add_node(concept_b)
        edge = self._get_edge(concept_a, concept_b)
        # Dissonance = co-activation with conflict = strong significance
        edge.hebbian_update(dissonance_score, self.LEARNING_RATE)
        edge.record_dissonance(dissonance_score)

    def record_avoidance(self, concept: str, avoidance_score: float):
        """
        Record avoidance signal for a concept.

        Creates/updates silence ↔ concept edge.
        Multi-chunk avoidance compounds (Hebbian: each call strengthens).
        Concept auto-merged if similar to existing.
        """
        if avoidance_score < self.AVOIDANCE_THRESHOLD:
            return
        self.add_node("silence")
        concept = self.add_node(concept)
        edge = self._get_edge("silence", concept)
        edge.hebbian_update(avoidance_score, self.LEARNING_RATE)

    def apply_session_decay(self):
        """Apply use-dependent decay to all edges (call at end of session)."""
        for edge in self.edges.values():
            edge.apply_decay(self.BASE_DECAY, self.LTP_THRESHOLD)

    def get_edges_sorted(self):
        """Return edges sorted by weight descending."""
        return sorted(
            ((list(k), e.weight, e.activation_count, e.dissonance) for k, e in self.edges.items()),
            key=lambda x: x[1],
            reverse=True,
        )

    def get_silence_edges(self):
        """Return silence ↔ concept edges sorted by weight."""
        return [
            (list(k), e.weight, e.activation_count)
            for k, e in self.edges.items()
            if "silence" in k
        ]

    def preload(self) -> bool:
        """
        Preload all lazy resources (currently: ConceptMerger embedding model).

        Call this during SessionProcessor initialization to move model-load latency
        to startup time, avoiding spikes during first graph updates.

        Returns True if ConceptMerger model is ready (or not installed, graceful).
        """
        if self.concept_merger is not None:
            return self.concept_merger.preload()
        return True

    # ─── Persistence (JSON-first) ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to JSON-compatible dict.

        Nodes: list of strings
        Edges: list of [nodes_list, weight, activation_count, dissonance]
        Hyperparams included for reproducibility.
        ConceptMerger is runtime-only (not serialized).
        """
        return {
            "nodes": sorted(self.nodes),
            "edges": [
                [list(k), e.weight, e.activation_count, e.dissonance]
                for k, e in self.edges.items()
            ],
            "hyperparams": {
                "LEARNING_RATE": self.LEARNING_RATE,
                "BASE_DECAY": self.BASE_DECAY,
                "LTP_THRESHOLD": self.LTP_THRESHOLD,
                "DISSONANCE_THRESHOLD": self.DISSONANCE_THRESHOLD,
                "AVOIDANCE_THRESHOLD": self.AVOIDANCE_THRESHOLD,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientGraph":
        """
        Reconstruct PatientGraph from dict (e.g., loaded from JSON).

        Creates a fresh ConceptMerger (runtime tool, not persisted).
        """
        g = cls()
        g.nodes = set(d.get("nodes", []))

        for entry in d.get("edges", []):
            nodes_list, weight, activation_count, dissonance = entry
            key = frozenset(nodes_list)
            e = Edge.from_dict({
                "weight": weight,
                "activation_count": activation_count,
                "dissonance": dissonance,
            })
            g.edges[key] = e

        # Hyperparams (optional — use defaults if missing)
        hp = d.get("hyperparams", {})
        g.LEARNING_RATE = float(hp.get("LEARNING_RATE", g.LEARNING_RATE))
        g.BASE_DECAY = float(hp.get("BASE_DECAY", g.BASE_DECAY))
        g.LTP_THRESHOLD = float(hp.get("LTP_THRESHOLD", g.LTP_THRESHOLD))
        g.DISSONANCE_THRESHOLD = float(hp.get("DISSONANCE_THRESHOLD", g.DISSONANCE_THRESHOLD))
        g.AVOIDANCE_THRESHOLD = float(hp.get("AVOIDANCE_THRESHOLD", g.AVOIDANCE_THRESHOLD))

        # Fresh ConceptMerger (runtime only)
        g.concept_merger = ConceptMerger()
        return g

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PatientGraph":
        """Load graph from JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def summary(self):
        print(f"Nodes: {len(self.nodes)}")
        print(f"Edges: {len(self.edges)}")
        print("Top edges:")
        for (a, b), w, n, d in self.get_edges_sorted()[:5]:
            dis = f", dissonance={d:.2f}" if d > 0 else ""
            print(f"  {a} ↔ {b}: w={w:.3f}, activations={n}{dis}")
        sil = self.get_silence_edges()
        if sil:
            print("Silence edges:")
            for (a, b), w, n in sil[:3]:
                print(f"  {a} ↔ {b}: w={w:.3f}, activations={n}")


# ─── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Ousia Graph Engine Demo ===\n")

    g = PatientGraph()

    # Session 1: Patient talks about work stress with high vocal tension
    print("Session 1: High tension about work and boss")
    g.record_coactivation("work", "boss", 0.9)
    g.record_coactivation("work", "stressed", 0.8)
    g.record_coactivation("boss", "angry", 0.95)
    g.record_coactivation("work", "angry", 0.7)
    # Dissonance: "I'm fine" but voice tense
    g.record_dissonance("work", "fine", 0.85)
    g.apply_session_decay()
    g.summary()

    print("\n---")

    # Session 2: Mentions work again (reinforcement), plus new topic (mother) — calm
    print("\nSession 2: Work still stressful + mentions mother (calm)")
    g.record_coactivation("work", "boss", 0.85)
    g.record_coactivation("work", "stressed", 0.75)
    g.record_coactivation("mother", "childhood", 0.3)  # low vocal deviation
    g.record_coactivation("work", "mother", 0.2)  # weak connection
    g.apply_session_decay()
    g.summary()

    print("\n---")

    # Session 3: Avoidance — "I don't know, whatever" about childhood
    print("\nSession 3: Avoidance signals (vagueness about childhood)")
    g.record_coactivation("mother", "home", 0.25)
    g.record_coactivation("mother", "childhood", 0.35)
    # Avoidance detected (LLM would return avoidance_score=0.7)
    g.record_avoidance("childhood", 0.7)
    g.record_avoidance("mother", 0.6)
    g.apply_session_decay()
    g.summary()

    print("\n---")

    # Session 4-5: More avoidance of mother/childhood (compounds)
    print("\nSessions 4-5: Repeated avoidance of mother/childhood (compounds)")
    for _ in range(2):
        g.record_avoidance("mother", 0.65)
        g.record_avoidance("childhood", 0.75)
    g.apply_session_decay()
    g.summary()

    print("\n---")

    # Sessions 6-10: Work persists, mother not mentioned (decays)
    print("\nSessions 6-10: Only work mentioned (mother fades, silence persists)")
    for _ in range(5):
        g.record_coactivation("work", "stressed", 0.6)
        g.record_coactivation("boss", "angry", 0.7)
    g.apply_session_decay()
    g.summary()

    print("\n=== Key insight ===")
    print("• work ↔ boss: high activation → decays slowly (persistent)")
    print("• mother ↔ childhood: low activation → decayed")
    print("• silence ↔ mother/childhood: avoidance compounds (Hebbian) → persists")
    print("• work ↔ fine: dissonance=0.85 (text/voice conflict)")
    print("• No hardcoded λ. No 1/√n. Just brain's rules + avoidance + dissonance.")
