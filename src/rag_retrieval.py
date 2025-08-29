#!/usr/bin/env python3
"""
Unified RAGatouille indexing + top-k retrieval on a JSON dataset.

- Per-sample indexing with RAGatouille (ColBERT under the hood)
- Top-k retrieval returning indices and scores
- Compatible with datasets that have per-sample contexts (e.g., "ctxs")

You can run in two phases or do both:
  1) prepare (build per-sample index)
  2) retrieve (compute top-k indices/scores)

Example:
  Prepare:
    python baselines/CacheBlend/src/rag_retrieval_single.py \
      --dataset inputs/musique_s.json \
      --action prepare \
      --model colbert-ir/colbertv2.0

  Retrieve:
    python baselines/CacheBlend/src/rag_retrieval_single.py \
      --dataset inputs/musique_s.json \
      --action retrieve \
      --top-k 8

  Both:
    python baselines/CacheBlend/src/rag_retrieval_single.py \
      --dataset inputs/musique_s.json \
      --action both \
      --top-k 8 \
      --model colbert-ir/colbertv2.0
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm
from ragatouille import RAGPretrainedModel


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _extract_texts_from_sample(sample: Dict[str, Any]) -> List[str]:
    """Extract a list of textual passages from a sample.
    Tries common fields; falls back to empty if none.
    """
    # Prefer MuSiQue-style ctxs
    if isinstance(sample.get("ctxs"), list):
        texts: List[str] = []
        for ch in sample["ctxs"]:
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            full = f"{title}\n{text}".strip() if title else text
            if full:
                texts.append(full.replace("\n", " "))
        return texts

    # Generic fallbacks
    if isinstance(sample.get("contents"), list):
        out = []
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
    for key in ("passage_id", "docid", "doc_id", "document_id", "id"):
        if key in hit:
            try:
                return int(hit[key])
            except Exception:
                pass
    return None


class ColbertRetrieval:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        model_id = self.config.get("model", "colbert-ir/colbertv2.0")
        r_text_index_key = self.config.get("r_text_index_key", "r_text_index")
        doc_key = self.config.get("doc_key", "doc_id")
        text_question_key = self.config.get("text_question_key", "question")
        dataset_name = self.config.get("dataset_name", "dataset")

        RAG = RAGPretrainedModel.from_pretrained(model_id)

        doc_index: Dict[str, str] = {}
        errors = 0

        for sample in tqdm(samples, desc="Indexing per-sample"):
            # Skip if index already present and path exists
            idx_path = sample.get(r_text_index_key)
            if isinstance(idx_path, str) and os.path.exists(os.path.join(idx_path, "pid_docid_map.json")):
                continue

            doc_val = str(sample.get(doc_key, ""))
            if not doc_val:
                # Use a fallback per-sample key if missing
                doc_val = str(sample.get("id", "sample"))

            if doc_val in doc_index:
                sample[r_text_index_key] = doc_index[doc_val]
                continue

            texts = _extract_texts_from_sample(sample)
            if not texts:
                sample[r_text_index_key] = ""
                continue

            index_name = f"{dataset_name}-{text_question_key}-{doc_val}"
            try:
                index_path = RAG.index(index_name=index_name, collection=texts)
                doc_index[doc_val] = index_path
                sample[r_text_index_key] = index_path
            except Exception as e:
                errors += 1
                sample[r_text_index_key] = ""
                if errors > max(1, len(samples) // 100):
                    print("Too many error cases. Exit process.")
                    raise
                print(f"Error processing {doc_val}: {e}")

        return samples

    def find_sample_top_k(self, sample: Dict[str, Any], top_k: int, page_id_key: str) -> Tuple[List[int], List[float]]:
        r_text_index_key = self.config.get("r_text_index_key", "r_text_index")
        text_question_key = self.config.get("text_question_key", "question")
        index_path = sample.get(r_text_index_key, "")

        pid_map_path = os.path.join(index_path, "pid_docid_map.json")
        if not os.path.exists(pid_map_path):
            print(f"Index not found for {pid_map_path}.")
            return [], []

        with open(pid_map_path, 'r') as f:
            pid_map_data = json.load(f)  # { passage_id(str) : docid(int-like) }

        # Build a consistent docid -> rank mapping
        unique_docids = list(dict.fromkeys(pid_map_data.values()))
        docid_to_rank = {val: idx for idx, val in enumerate(unique_docids)}
        # Map passage_id -> page rank
        pid_to_page = {int(k): docid_to_rank[v] for k, v in pid_map_data.items() if str(k).isdigit()}

        query = str(sample.get(text_question_key, ""))
        if not query:
            return [], []

        RAG = RAGPretrainedModel.from_index(index_path)
        results = RAG.search(query, k=max(top_k, len(pid_to_page) or top_k))

        top_indices: List[int] = []
        top_scores: List[float] = []

        # Prefer passage_id if available, else use docid variants
        for hit in results:
            pid_or_doc = _get_hit_doc_id(hit)
            if pid_or_doc is None:
                continue
            idx = pid_to_page.get(pid_or_doc)
            if idx is None:
                # If the returned id is a docid, map via docid_to_rank
                idx = docid_to_rank.get(pid_or_doc)
            if idx is None:
                continue
            score = float(hit.get("score", 0.0))
            top_indices.append(idx)
            top_scores.append(score)
            if len(top_indices) >= top_k:
                break

        if page_id_key in sample and isinstance(sample[page_id_key], list):
            allowed = set(int(x) for x in sample[page_id_key])
            filtered_idx: List[int] = []
            filtered_scores: List[float] = []
            for idx, sc in zip(top_indices, top_scores):
                if idx in allowed:
                    filtered_idx.append(idx)
                    filtered_scores.append(sc)
            return filtered_idx[:top_k], filtered_scores[:top_k]

        return top_indices[:top_k], top_scores[:top_k]

    def run(self, dataset_path: str, action: str = "both") -> str:
        # Load dataset
        with open(dataset_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]

        # Attach dataset_name if not provided
        if not self.config.get("dataset_name"):
            base = os.path.splitext(os.path.basename(dataset_path))[0]
            self.config["dataset_name"] = base

        # Keys
        r_text_index_key = self.config.get("r_text_index_key", "r_text_index")
        r_text_key = self.config.get("r_text_key", "retrieved_indices")
        page_id_key = self.config.get("page_id_key", "page_ids")
        top_k = int(self.config.get("top_k", 8))

        # Prepare
        if action in ("both", "prepare"):
            data = self.prepare(data)

        # Retrieve
        if action in ("both", "retrieve"):
            for sample in tqdm(data, desc="Retrieving top-k"):
                idxs, scores = self.find_sample_top_k(sample, top_k=top_k, page_id_key=page_id_key)
                sample[r_text_key] = idxs
                sample[r_text_key + "_score"] = scores

        # Save next to dataset
        out_path = self.config.get("out_path", dataset_path.replace('.json', f'_rag_{action}_k{top_k}.json'))
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path}")
        return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Single-file RAG indexing + retrieval")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON")
    p.add_argument("--action", choices=["prepare", "retrieve", "both"], default="both")
    p.add_argument("--model", default="colbert-ir/colbertv2.0", help="HF model id for RAG")
    p.add_argument("--top-k", type=int, default=8)
    # Key names (customize per dataset)
    p.add_argument("--doc-key", default="doc_id")
    p.add_argument("--text-question-key", default="question")
    p.add_argument("--r-text-index-key", default="r_text_index")
    p.add_argument("--r-text-key", default="retrieved_indices")
    p.add_argument("--page-id-key", default="page_ids")
    p.add_argument("--out_path", default="")
    args = p.parse_args()

    cfg = {
        "model": args.model,
        "top_k": args.top_k,
        "doc_key": args.doc_key,
        "text_question_key": args.text_question_key,
        "r_text_index_key": args.r_text_index_key,
        "r_text_key": args.r_text_key,
        "page_id_key": args.page_id_key,
    }

    runner = ColbertRetrieval(cfg)
    runner.run(args.dataset, action=args.action)


if __name__ == "__main__":
    main()


