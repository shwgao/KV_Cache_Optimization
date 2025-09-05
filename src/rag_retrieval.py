#!/usr/bin/env python3
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ragatouille import RAGPretrainedModel


# --------------------------- config ---------------------------

@dataclass
class RetrievalConfig:
    # Model & dataset
    model_id: str = "colbert-ir/colbertv2.0"
    dataset_name: str = "pipeline_dataset"

    # Key names inside each sample
    r_text_index_key: str = "r_text_index"        # where per-sample index path is stored
    doc_key: str = "doc_id"                        # optional, used to name per-sample index
    question_key: str = "question"                 # text query field
    retrieved_key: str = "retrieved_indices"       # output indices
    page_id_key: str = "page_ids"                  # optional allow-list of page indices

    # Defaults
    top_k: int = 5


# ---------------------- small utilities ----------------------

def _extract_texts_from_sample(sample: Dict[str, Any]) -> List[str]:
    """
    Extract textual passages for indexing.
    - Prefer MuSiQue-style: sample["ctxs"] with {title, text}
    - Fallbacks: sample["contents"] as list[str|dict]
    """
    if isinstance(sample.get("ctxs"), list):
        texts: List[str] = []
        for ch in sample["ctxs"]:
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                texts.append(full.replace("\n", " "))
        return texts

    if isinstance(sample.get("contents"), list):
        out: List[str] = []
        for it in sample["contents"]:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s.replace("\n", " "))
            elif isinstance(it, dict):
                s = (it.get("text") or it.get("content") or "").strip()
                if s:
                    out.append(s.replace("\n", " "))
        return out

    return []


def _get_hit_doc_id(hit: Dict[str, Any]) -> Optional[int]:
    """Try common id fields returned by RAGatouille/ColBERT."""
    for key in ("passage_id", "docid", "doc_id", "document_id", "id"):
        if key in hit:
            try:
                return int(hit[key])
            except Exception:
                pass
    return None


def _pid_map_ok(index_path: str) -> bool:
    """Check that per-sample index directory looks valid."""
    if not isinstance(index_path, str) or not index_path:
        return False
    pid_map = os.path.join(index_path, "pid_docid_map.json")
    return os.path.exists(pid_map)


# --------------------- main retriever class -------------------

class ColbertRetrieval:
    """
    Minimal ColBERT retriever using RAGatouille under the hood.

    Typical usage from pipeline.py:
        retr = ColbertRetrieval(RetrievalConfig(top_k=cfg.cache.max_gpu_chunks))
        samples = retr.prepare(samples)                  # add per-sample index paths
        samples = retr.retrieve(samples)                 # add top-k indices + scores
        # or per-sample:
        idxs, scores = retr.find_sample_top_k(sample, top_k=16)
    """

    def __init__(self, config: RetrievalConfig | Dict[str, Any] = RetrievalConfig()):
        if isinstance(config, dict):
            self.config = RetrievalConfig(**config)
        else:
            self.config = config

        # Lazy handles to avoid repeated loads
        self._indexer: Optional[RAGPretrainedModel] = None

    # -------- public API --------

    def prepare(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build per-sample ColBERT indices (if missing) and store the path in r_text_index_key.
        Idempotent: skips samples that already have a valid index.
        """
        cfg = self.config
        RAG = self._get_indexer()  # single shared indexer

        # Map to reuse an index if the same document key appears multiple times
        cached_index_path_for_doc: Dict[str, str] = {}

        for sample in samples:
            # if already valid, skip
            existing = sample.get(cfg.r_text_index_key)
            if _pid_map_ok(existing):
                continue

            # choose a stable per-sample name
            doc_val = str(sample.get(cfg.doc_key, "") or sample.get("id", "sample"))

            if doc_val in cached_index_path_for_doc:
                sample[cfg.r_text_index_key] = cached_index_path_for_doc[doc_val]
                continue

            texts = _extract_texts_from_sample(sample)
            if not texts:
                sample[cfg.r_text_index_key] = ""
                continue

            index_name = f"{cfg.dataset_name}-{cfg.question_key}-{doc_val}"
            try:
                index_path = RAG.index(index_name=index_name, collection=texts)
            except Exception as e:
                # On failure, mark empty and continue; caller can decide how to handle
                sample[cfg.r_text_index_key] = ""
                continue

            cached_index_path_for_doc[doc_val] = index_path
            sample[cfg.r_text_index_key] = index_path

        return samples

    def retrieve(self, samples: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Compute top-k page indices (+scores) for each sample.
        Writes in-place:
            sample[cfg.retrieved_key] = List[int]
            sample[cfg.retrieved_key + "_score"] = List[float]
        """
        cfg = self.config
        k = int(top_k or cfg.top_k)

        for sample in samples:
            idxs, scores = self.find_sample_top_k(sample, top_k=k)
            sample[cfg.retrieved_key] = idxs
            sample[cfg.retrieved_key + "_score"] = scores

        return samples

    def find_sample_top_k(self, sample: Dict[str, Any], top_k: Optional[int] = None) -> Tuple[List[int], List[float]]:
        """
        Return (indices, scores) for a single sample.
        - Respects page allow-list if sample contains cfg.page_id_key.
        - Returns [] if index not found or question missing.
        """
        cfg = self.config
        k = int(top_k or cfg.top_k)

        index_path = sample.get(cfg.r_text_index_key, "")
        if not _pid_map_ok(index_path):
            return [], []

        # Load pid -> docid mapping
        pid_map_path = os.path.join(index_path, "pid_docid_map.json")
        with open(pid_map_path, "r") as f:
            pid_map_data = json.load(f)  # { passage_id(str) : docid (int-like) }

        # Build docid -> rank and pid -> page
        unique_docids = list(dict.fromkeys(pid_map_data.values()))
        docid_to_rank = {val: idx for idx, val in enumerate(unique_docids)}
        pid_to_page = {int(k): docid_to_rank[v] for k, v in pid_map_data.items() if str(k).isdigit()}

        query = str(sample.get(cfg.question_key, "")).strip()
        if not query:
            return [], []

        RAG = RAGPretrainedModel.from_index(index_path)
        results = RAG.search(query, k=max(k, len(pid_to_page) or k))

        top_indices: List[int] = []
        top_scores: List[float] = []

        for hit in results:
            pid_or_doc = _get_hit_doc_id(hit)
            if pid_or_doc is None:
                continue
            page_idx = pid_to_page.get(pid_or_doc)
            if page_idx is None:
                # If ColBERT returns docid directly, map via docid_to_rank
                page_idx = docid_to_rank.get(pid_or_doc)
            if page_idx is None:
                continue

            score = float(hit.get("score", 0.0))
            top_indices.append(page_idx)
            top_scores.append(score)
            if len(top_indices) >= k:
                break

        # Optional allow-list filtering (e.g., some datasets provide valid page ids)
        if cfg.page_id_key in sample and isinstance(sample[cfg.page_id_key], list):
            allowed = set(int(x) for x in sample[cfg.page_id_key])
            f_idx, f_sc = [], []
            for i, sc in zip(top_indices, top_scores):
                if i in allowed:
                    f_idx.append(i)
                    f_sc.append(sc)
            return f_idx[:k], f_sc[:k]

        return top_indices[:k], top_scores[:k]

    # -------- internals --------

    def _get_indexer(self) -> RAGPretrainedModel:
        """Create (once) and reuse a single pre-trained model handle for indexing."""
        if self._indexer is None:
            self._indexer = RAGPretrainedModel.from_pretrained(self.config.model_id)
        return self._indexer
