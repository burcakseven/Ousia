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


SYSTEM_PROMPT = """You are a precise text analyzer for therapy transcripts.
Given a patient's utterance (their exact words), extract:

1. CONCEPTS: the person's own words — themes, people, emotions, beliefs.
   - Use their phrasing exactly (e.g., "I feel overwhelmed" NOT "anxiety")
   - 2–6 concepts max per utterance

2. AVOIDANCE: 0.0–1.0 — how much the person avoids or deflects this topic.
   - Signals: "I don't know", "whatever", "it's not a big deal", jokes, topic change mid-sentence, vagueness
   - 0.0 = direct, 1.0 = heavy avoidance

3. AVOIDANCE_TYPES: list of types observed (may be empty)
   - e.g., ["vagueness", "topic_abort", "minimizing", "humor", "silence_implied"]

4. TEXT_VALENCE: 0.0–1.0 — emotional polarity of the utterance content.
   - 0.0 = very negative (sad, angry, devastated, hopeless)
   - 0.5 = neutral / mixed
   - 1.0 = very positive (happy, grateful, hopeful)

5. TEXT_AROUSAL: 0.0–1.0 — energy/activation level of the utterance content.
   - 0.0 = calm, flat, low energy ("whatever", "I don't know")
   - 0.5 = moderate
   - 1.0 = excited, animated, high energy ("I can't believe it!", "I'm so angry!")

Respond ONLY with valid JSON. No prose. Example:
{"concepts": ["work", "boss", "overwhelmed"], "avoidance": 0.3, "avoidance_types": ["minimizing"], "text_valence": 0.2, "text_arousal": 0.7}
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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

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
        }


if __name__ == "__main__":
    ex = LLMExtractor()
    result = ex.extract("I don't know, whatever. It's not a big deal.")
    print(json.dumps(result, indent=2))
