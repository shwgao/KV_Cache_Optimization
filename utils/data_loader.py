import json
from typing import List, Tuple, Optional

def load_rag_chunks(chunks_json: str) -> Tuple[List[str], Optional[str], Optional[List[str]]]:
    with open(chunks_json, "r") as f:
        data = json.load(f)

    # Case 1: single sample with ctxs
    if isinstance(data, dict) and "ctxs" in data:
        chunks = [c.get("text", "").strip() for c in data["ctxs"] if c.get("text")]
        return chunks, data.get("question"), data.get("answers")

    # Case 2: list of samples with ctxs
    if isinstance(data, list) and data and isinstance(data[0], dict) and "ctxs" in data[0]:
        first = data[0]
        chunks = [c.get("text", "").strip() for c in first["ctxs"] if c.get("text")]
        return chunks, first.get("question"), first.get("answers")

    # Case 3: dict with "chunks"
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"], None, None

    # Case 4: plain list of strings
    if isinstance(data, list):
        return [str(c) for c in data], None, None

    # Case 5: fallback
    return [str(data)], None, None
