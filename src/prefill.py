# cb_prefill.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ----------------------------
# Public helper: build_chunk_ids
# ----------------------------
def build_chunk_ids(
    tokenizer: AutoTokenizer,
    doc_prompts: List[str],
    q_prompt: str,
    prefix_prompt: str,
) -> Tuple[List[List[int]], List[int], Dict[str, Any]]:
    """
    Mirrors the user's snippet exactly (same brittle sentinels and slicing math).
    Returns:
        doc_chunk_ids: segments where:
            [0] is s_start_full + prefix_prompt
            [1..N] are per-doc segments prefixed with s_start (empty list here)
            [N+1] is s_start + q_ids + s_end
        q_ids: token ids for the question (without BOS)
        meta: dict with fields used by fuser.collect() and stitch_input_ids()
    """
    # Hard-coded tokens (same as your snippet; model/tokenizer-specific!)
    S_START_FULL_HEAD = [733, 16289, 28793]  # <s> USER :
    S_END             = [733, 28748, 16289, 28793]  # <s> ASSISTANT :
    S_START           = []  # intentionally empty (as in your snippet)

    # Tokenize
    s_start_full = S_START_FULL_HEAD + tokenizer.encode(prefix_prompt)[1:]
    s_start_len  = len(s_start_full) + 1  # +1 due to vLLM BOS-like offset

    s_start_1_len = len(S_START) + 1

    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids         = tokenizer.encode(q_prompt)[1:]

    # Assemble segments
    doc_chunk_ids = [S_START + ids for ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [S_START + q_ids + S_END]

    # meta for downstream steps
    meta = {
        "s_start_len":  s_start_len,
        "s_start_1_len": s_start_1_len,
        "s_end":         S_END,
        "last_len":      len(q_ids) + len(S_END),  # FIXED (your original had len([q_ids+s_end])==1)
        "S_START":       S_START,
        "S_END":         S_END,
        "S_START_FULL_HEAD": S_START_FULL_HEAD,
    }
    return doc_chunk_ids, q_ids, meta


# ----------------------------
# CacheBlend-style Fuser
# ----------------------------
@dataclass
class FusedKV:
    kv: List[List[torch.Tensor]]  # per-layer [K, V]; shapes [S, ...]
    total_prefix_tokens: int      # tokens represented in the fused KV (approx)


class CacheBlendFuser:
    """
    CacheBlend-style prefill-only module:

    - Prefills each segment with vLLM (max_tokens=1) to populate per-layer hack_kv.
    - Slices and concatenates the K/V across segments into a single fused prefix KV,
      exactly following your snippet's indices.
    - Optionally SAVES per-segment K/V to disk: {save_dir}/{idx}/keys.pt, values.pt (+ metadata.json)
      so your scheduler can load/merge them later.
    - Can decode with the fused cache (so your current cb_fuse branch works unchanged).

    NOTE: relies on vLLM non-public fields (hack_kv, old_kvs, cache_fuse_metadata).
    """

    def __init__(
        self,
        model_id: str,
        gpu_mem_util: float = 0.5,
        save_cache_dir: Optional[str] = None,  # if provided, collect() will persist per-chunk KVs here
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.model_id = model_id
        self.llm      = LLM(model=model_id, gpu_memory_utilization=gpu_mem_util)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.llm.set_tokenizer(self.tokenizer)

        self._model  = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model
        self._layers = self._model.layers
        self._L      = len(self._layers)

        # Non-public control
        self._cache_meta = self._model.cache_fuse_metadata

        # Storage for fused KV from last collect()
        self._fused: Optional[FusedKV] = None

        # Optional on-disk save root
        self.save_cache_dir = save_cache_dir  # set per-sample in pipeline if you want to persist

    # ---------------- internals ----------------

    def _prefill_segment(self, seg_ids: List[int]) -> None:
        """Force a prefill step so vLLM computes KV for this segment (no decoding)."""
        sp = SamplingParams(temperature=0, max_tokens=1)
        self.llm.generate([self.tokenizer.decode(seg_ids)], sp)

    def _slice_layer_kv(self, layer_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read [K,V] from non-public hook and slice; returns clones on current device."""
        pkv = self._layers[layer_idx].self_attn.hack_kv  # tuple (K, V)
        K = pkv[0][start:end].clone()
        V = pkv[1][start:end].clone()
        return K, V

    def _save_chunk_kv(
        self,
        out_dir: str,
        seg_idx: int,
        Ks: List[torch.Tensor],
        Vs: List[torch.Tensor],
        sample_id: str,
    ) -> None:
        """
        Save per-segment K/V as:
            {out_dir}/{seg_idx}/keys.pt  (shape: [L, H, S, D])
            {out_dir}/{seg_idx}/values.pt
            {out_dir}/{seg_idx}/metadata.json
        """
        os.makedirs(out_dir, exist_ok=True)
        chunk_dir = os.path.join(out_dir, str(seg_idx))
        os.makedirs(chunk_dir, exist_ok=True)

        # Pack per-layer tensors to [L, H, S, D]
        # We assume Ks[L] has shape [S, H, D] or [S, ...]; the snippet stores [S, ...]
        # We standardize to [L, H, S, D] by permuting [S, H, D] -> [H, S, D] and stacking on first dim.
        stackK = []
        stackV = []
        for K, V in zip(Ks, Vs):
            # K/V are [S, ..., ...]; in vLLM Mistral they are [S, n_heads, head_dim]
            if K.dim() == 3:  # [S, H, D]
                K4 = K.permute(1, 0, 2).contiguous()  # [H, S, D]
                V4 = V.permute(1, 0, 2).contiguous()  # [H, S, D]
            else:
                # If shape unexpected, try best-effort keep as-is along (H, S, D) = last 3 dims
                K4 = K
                V4 = V
            stackK.append(K4.cpu())
            stackV.append(V4.cpu())
        K_out = torch.stack(stackK, dim=0)  # [L, H, S, D]
        V_out = torch.stack(stackV, dim=0)  # [L, H, S, D]

        torch.save(K_out, os.path.join(chunk_dir, "keys.pt"))
        torch.save(V_out, os.path.join(chunk_dir, "values.pt"))

        meta = {
            "chunk_id": f"{sample_id}_chunk{seg_idx}",
            "shapes": {
                "keys": list(K_out.shape),
                "values": list(V_out.shape),
            },
            "dtype": str(K_out.dtype).replace("torch.", ""),
        }
        with open(os.path.join(chunk_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---------------- public API ----------------

    def collect(
        self,
        doc_chunk_ids: List[List[int]],
        meta: Dict[str, Any],
        *,
        sample_id: str = "sample",
        save_dir: Optional[str] = None,
        save_per_chunk: bool = True,
        include_question_in_fused: bool = True,
    ) -> Dict[str, Any]:
        """
        CacheBlend-style COLLECT (no decoding):
        - Prefill each segment (max_tokens=1)
        - Slice and concatenate K/V into fused prefix KV (optionally including the question tail)
        - Optionally SAVE each segment's KV to disk (so your scheduler can load/merge later)

        Args:
            doc_chunk_ids: segments from build_chunk_ids()
            meta: dict returned by build_chunk_ids()
            sample_id: used only in metadata.json (defaults to "sample")
            save_dir: if provided (or if self.save_cache_dir is set), writes per-chunk KVs here.
            save_per_chunk: if True, saves each segment separately as {idx}/keys.pt, values.pt
            include_question_in_fused: mirror your original loop (includes last segment in fused KV)

        Returns: small summary dict (and stores fused KV internally for decode_with_fused)
        """
        s_start_len  = int(meta["s_start_len"])
        s_start_1_len = int(meta["s_start_1_len"])
        last_len      = int(meta["last_len"])

        out_root = save_dir or self.save_cache_dir  # may be None (then we don't save)
        if out_root:
            os.makedirs(out_root, exist_ok=True)

        # Turn on cache collection mode
        self._cache_meta["collect"] = True
        self._cache_meta["check"]   = False

        fused_KV: List[List[torch.Tensor]] = []
        total_prefix_tokens = 0

        # For saving per-chunk: we need to keep per-layer Ks/Vs for the exact slice we append
        for seg_idx, seg_ids in enumerate(doc_chunk_ids):
            self._prefill_segment(seg_ids)

            # Slicing policy (exactly the same as your code):
            if seg_idx == 0:
                start, end = 0, s_start_len
            else:
                start, end = s_start_1_len, len(seg_ids) + 1

            # If you want to exclude the question from fused KV, skip the last segment here
            is_last = (seg_idx == len(doc_chunk_ids) - 1)
            take_for_fused = (include_question_in_fused or (not is_last))

            # Accumulate fused KV and optionally save per-chunk slice
            perL_Ks: List[torch.Tensor] = []
            perL_Vs: List[torch.Tensor] = []

            for L in range(self._L):
                K, V = self._slice_layer_kv(L, start, end)  # [S, H, D]
                perL_Ks.append(K)
                perL_Vs.append(V)

                if take_for_fused:
                    if seg_idx == 0:
                        # initialize fused KV list with first segment
                        if seg_idx == 0 and L == 0:
                            fused_KV = []  # ensure fresh
                            for _ in range(self._L):
                                fused_KV.append([None, None])  # placeholders
                    # append to fused
                    if fused_KV[L][0] is None:
                        fused_KV[L][0] = K
                        fused_KV[L][1] = V
                    else:
                        fused_KV[L][0] = torch.cat((fused_KV[L][0], K), dim=0)
                        fused_KV[L][1] = torch.cat((fused_KV[L][1], V), dim=0)

            if take_for_fused:
                total_prefix_tokens += (end - start)

            # Persist per-chunk (scheduler compatibility)
            if out_root and save_per_chunk:
                self._save_chunk_kv(out_root, seg_idx, perL_Ks, perL_Vs, sample_id=sample_id)

            # keep model.old_kvs in sync (parity with your snippet)
            self._model.old_kvs = fused_KV if fused_KV else []

        # Turn off collection mode
        self._cache_meta["collect"] = False

        self._fused = FusedKV(kv=fused_KV if fused_KV else [], total_prefix_tokens=total_prefix_tokens)

        # Write a small top-level summary if saving
        if out_root:
            summary = {
                "sample_id": sample_id,
                "num_segments": len(doc_chunk_ids),
                "total_prefix_tokens": total_prefix_tokens,
                "include_question_in_fused": include_question_in_fused,
                "last_len": last_len,
                "layers": self._L,
            }
            with open(os.path.join(out_root, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        return {
            "total_prefix_tokens": total_prefix_tokens,
            "layers": self._L,
            "saved_to": out_root,
            "last_len": last_len,
        }

    def stitch_input_ids(self, doc_chunk_ids: List[List[int]], meta: Dict[str, Any]) -> List[int]:
        """
        Flatten segments into a single input_ids list (same as your code).
        """
        s_start_1_len = int(meta["s_start_1_len"])
        input_ids: List[int] = []
        for i, seg in enumerate(doc_chunk_ids):
            if i == 0:
                temp = seg
            else:
                temp = seg[s_start_1_len - 1 :]
            input_ids += temp
        return input_ids

    def decode_with_fused(
        self,
        input_ids: List[int],
        suffix_len: int,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Optional helper to generate using the fused KV (so your existing cb_fuse branch works).
        """
        if self._fused is None or not self._fused.kv:
            raise RuntimeError("No fused KV collected. Call collect(...) first.")

        # Inject fused KV and set cache-fuse flags
        self._model.old_kvs = self._fused.kv
        self._cache_meta["check"]      = True
        self._cache_meta["collect"]    = False
        self._cache_meta["suffix_len"] = int(suffix_len)

        prompt = self.tokenizer.decode(input_ids)
        sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

        out = self.llm.generate([prompt], sp)[0]
        text = out.outputs[0].text
        ttft = out.metrics.first_token_time - out.metrics.first_scheduled_time
        # tokens/sec is not directly given; omit or compute from durations if you prefer

        return {
            "text": text,
            "ttft_s": float(ttft),
            "decoded_tokens": len(out.outputs[0].token_ids),
        }
