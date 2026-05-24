"""
LLM Extractor — text-only → concepts, avoidance, text_valence, text_arousal.

Single LLM call per utterance. Outputs valence (positive↔negative) and arousal (calm↔excited)
from text content. Voice analysis is separate (VocalAnalyzer); dissonance is computed post-hoc
in SessionProcessor as |text_arousal − vocal_salience|.

Uses an OpenAI-compatible API for text analysis.
"""

import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env


SYSTEM_PROMPT = """You are a highly skilled Psychological Insight Engine for clinical session analysis.
Your goal is to look BEYOND the raw surface words (Ousia: the essence) and identify the underlying psychological constructs.

Given a transcript utterance, extract:

1. CONCEPTS (Psychological Synthesis): 
   - Instead of literal keywords, output abstract psychological themes, defense mechanisms, or schemas.
   - e.g., instead of "work stress", use "Performance-Based Self-Worth" or "Chronic Burnout".
   - e.g., instead of "mom", use "Parental Enmeshment" or "Maternal Conflict".
   - Identify: Defense Mechanisms (e.g., Intellectualization, Projection), Core Beliefs (e.g., "I am unlovable"), or Emotional Patterns.
   - 2-5 high-level concepts max.

2. AVOIDANCE (0.0–1.0): 
   - 0.0 = direct/authentic, 1.0 = heavy deflection.
   - Look for: intellectualization, humor, vagueness, topic-shifting, minimizing.

3. AVOIDANCE_TYPES: Technical names for the avoidance (e.g., ["minimizing", "rationalization", "humorous_deflection"]).

4. TEXT_VALENCE (0.0–1.0): Polarity of the underlying sentiment.

5. TEXT_AROUSAL (0.0–1.0): Psychological activation/energy level.

6. INSIGHT: A single concise sentence (max 15 words) explaining the psychological "move" the person is making in this utterance.

Respond ONLY with valid JSON. No prose.
Example:
{
  "concepts": ["Externalized Responsibility", "Passive-Aggressive Posturing"],
  "avoidance": 0.7,
  "avoidance_types": ["rationalization"],
  "text_valence": 0.3,
  "text_arousal": 0.6,
  "insight": "Deflecting guilt by attributing responsibility to external circumstances."
}
"""


class LLMExtractor:
    def __init__(
        self,
        model: str = "grok-3",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        model: LLM model name (e.g., grok-3, gpt-4o)
        base_url: API base URL
        api_key: explicit key, else LLM_API_KEY env var
        """
        llm_key = os.getenv("LLM_API_KEY")

        if base_url is None:
            base_url = "https://api.x.ai/v1"
            api_key = api_key or llm_key

        if not api_key:
            # Graceful fallback: no API key — extract() will return neutral signals
            self.client = None
            self.model = model
            return

        # Tracking metrics
        self.total_calls = 0
        self.failed_calls = 0

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def extract(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Single LLM call for one utterance (text-only).

        context: optional previous sentence for continuity (helps detect topic abortion).
        Returns concepts, avoidance, text_valence, text_arousal.

        If no API key was provided at init, returns neutral signals (graceful fallback).
        """
        # Graceful fallback if no client (no API key)
        if self.client is None:
            return {
                "concepts": [],
                "avoidance": 0.0,
                "avoidance_types": [],
                "text_valence": 0.5,
                "text_arousal": 0.5,
            }

        # Build user prompt
        user_prompt = f"Utterance: {text}"
        if context:
            user_prompt = f"Previous context: {context}\n\nUtterance: {text}"

        self.total_calls += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self.failed_calls += 1
            raise e

        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"concepts": [], "avoidance": 0.0, "avoidance_types": [], "text_valence": 0.5, "text_arousal": 0.5}

        # Normalize
        return {
            "concepts": data.get("concepts", []),
            "avoidance": float(data.get("avoidance", 0.0)),
            "avoidance_types": data.get("avoidance_types", []),
            "text_valence": float(data.get("text_valence", 0.5)),
            "text_arousal": float(data.get("text_arousal", 0.5)),
            "insight": data.get("insight", "")
        }


if __name__ == "__main__":
    ex = LLMExtractor()
    result = ex.extract("I don't know, whatever. It's not a big deal.")
    print(json.dumps(result, indent=2))
