"""
extractor.py
JSON Fact Extractor — AlphaLens Literature Agent
Calls Claude (claude-sonnet-4-6) on each retrieved chunk to extract
structured quantitative facts. Logs failures but continues processing.
"""

import json
import os
from typing import List, Dict

import anthropic
from .prompts import build_extraction_prompt


def extract_facts_from_chunks(
    chunks: List[Dict],
    model: str = "claude-sonnet-4-6",
) -> List[Dict]:
    """
    For each retrieved chunk, call the LLM and extract structured facts.
    Returns flat list of fact dicts.
    Skips chunks where LLM returns invalid JSON (logs warning).
    """
    client = anthropic.Anthropic()
    all_facts = []

    for chunk in chunks:
        prompt = build_extraction_prompt(chunk["text"])
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            raw_text = message.content[0].text.strip()
            # Strip markdown fences if LLM accidentally includes them
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            facts = json.loads(raw_text)
            if isinstance(facts, list):
                for fact in facts:
                    fact["source_chunk_id"] = chunk["chunk_id"]
                all_facts.extend(facts)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] Extraction failed for {chunk['chunk_id']}: {e}")

    return all_facts


def save_facts(
    facts: List[Dict],
    path: str = "outputs/literature_facts.json",
) -> None:
    """Persist extracted facts to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(facts, f, indent=2)
    print(f"[EXTRACTOR] Saved {len(facts)} facts → {path}")
