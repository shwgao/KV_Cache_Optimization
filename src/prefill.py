# prefill_cb_vllm.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Tuple, Optional

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --------------- helpers ---------------

def build_chunk_ids(tokenizer, doc_prompts: List[str], q_prompt: str,
                    prefix_prompt: str) -> Tuple[List[List[int]], List[int], Dict[str, int]]:
    """
    Returns:
      doc_chunk_ids: list of token id lists (one per passage, with s_start applied)
      q_ids: token ids of question segment (w/o s_start)
      meta: dictionary including lengths we need for slicing
    """
    s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
    s_start     = []  # your original code
    s_end       = [733, 28748, 16289, 28793]

    s_start_len   = len(s_start_full) + 1     # +1 b/c vLLM inserts a BOS-like offset in hack_kv
    s_start_1_len = len(s_start) + 1
    s_end_len     = len(s_end)

    doc_chunk_ids = [s_start + tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids         = tokenizer.encode(q_prompt)[1:]

    # Layout: [ s_start_full ] + [ s_start + doc_i ]*N + [ s_start + q_ids + s_end ]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids + [s_start + q_ids + s_end]


    last_len = len(q_ids) + len(s_end)

    meta = dict(
        s_start_len=s_start_len,
        s_start_1_len=s_start_1_len,
        s_end_len=s_end_len,
        last_len=last_len
    )
    return doc_chunk_ids, q_ids, meta


class CacheBlendFuser:
    """
    vLLM-based prefill/fuse engine compatible with the CacheBlend code path you pasted.
    - collect(): builds old_kvs by concatenating per-chunk hack_kv slices
    - decode_with_fused(): runs generation with check=True, using old_kvs
    - append_chunk(): later, you can append one more chunk to the existing old_kvs
    """

    def __init__(self, model_id: str, gpu_mem_util=0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, gpu_memory_utilization=gpu_mem_util)
        self.llm.set_tokenizer(self.tokenizer)
        self._cache_meta = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
        self._layers = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        self.num_layers = len(self._layers)
        self.old_kvs = None  # set after collect()

    # -- internal: one short forward to collect hack_kv for a chunk
    def _collect_one(self, token_ids: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # run a 1-token generation to force the model to build K/V for the prefix
        sp = SamplingParams(temperature=0, max_tokens=1)
        # Decode without cleanup to preserve exact tokenization; avoids off-by-one when re-encoding
        self.llm.generate([self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)], sp)
        # read hack_kv per layer
        kvs = []
        for j in range(self.num_layers):
            k, v = self._layers[j].self_attn.hack_kv  # shape [T, ...]
            kvs.append((k.clone(), v.clone()))
        return kvs

    def collect(self, doc_chunk_ids: List[List[int]], meta: Dict[str, int]) -> Dict[str, Any]:
        s_start_len   = meta["s_start_len"]
        s_start_1_len = meta["s_start_1_len"]

        self._cache_meta["collect"] = True
        self._cache_meta["check"]   = False

        chunk_past = None
        for i, ids in enumerate(doc_chunk_ids):
            kvs = self._collect_one(ids)
            # slice like your code
            if i == 0:
                # keep only the s_start_full segment
                sliced = [(k[:s_start_len].contiguous(), v[:s_start_len].contiguous()) for (k, v) in kvs]
                chunk_past = [[k, v] for (k, v) in sliced]  # becomes a list per layer
            else:
                # strip the initial s_start (short) window
                sliced = [(k[s_start_1_len:len(ids)+1].contiguous(),
                           v[s_start_1_len:len(ids)+1].contiguous()) for (k, v) in kvs]
                # concat on T dimension
                for j in range(self.num_layers):
                    chunk_past[j][0] = torch.cat((chunk_past[j][0], sliced[j][0]), dim=0)
                    chunk_past[j][1] = torch.cat((chunk_past[j][1], sliced[j][1]), dim=0)

        self.old_kvs = chunk_past
        # install into model
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = self.old_kvs
        return {"kv_layers": self.num_layers, "total_len": int(self.old_kvs[0][0].shape[0])}

    def append_chunk(self, chunk_ids: List[int], meta: Dict[str, int]):
        """
        Append one more chunk under the current left context: re-collect for this chunk and
        concatenate onto self.old_kvs
        """
        assert self.old_kvs is not None, "Call collect() once before append_chunk()."
        s_start_1_len = meta["s_start_1_len"]

        # collect for the new chunk
        kvs = self._collect_one(chunk_ids)
        sliced = [(k[s_start_1_len:len(chunk_ids)+1].contiguous(),
                   v[s_start_1_len:len(chunk_ids)+1].contiguous()) for (k, v) in kvs]

        for j in range(self.num_layers):
            self.old_kvs[j][0] = torch.cat((self.old_kvs[j][0], sliced[j][0]), dim=0)
            self.old_kvs[j][1] = torch.cat((self.old_kvs[j][1], sliced[j][1]), dim=0)

        # re-install
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = self.old_kvs

    def decode_with_fused(self, input_ids: List[int], suffix_len: int,
                          max_new_tokens: int = 32) -> Dict[str, Any]:
        """
        Use the fused old_kvs to generate with low TTFT.
        """
        assert self.old_kvs is not None, "collect() must be called before decode_with_fused()."
        self._cache_meta["check"] = True
        self._cache_meta["collect"] = False
        self._cache_meta["suffix_len"] = suffix_len

        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        # Decode without cleanup to preserve exact tokenization across boundaries
        out = self.llm.generate([self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)], sp)[0]
        text = out.outputs[0].text
        ttft = out.metrics.first_token_time - out.metrics.first_scheduled_time
        return {"text": text, "ttft_s": float(ttft)}

    # Optional: build the input_ids the same way as your script
    def stitch_input_ids(self, doc_chunk_ids: List[List[int]], meta: Dict[str, int]) -> List[int]:
        s_start_1_len = meta["s_start_1_len"]
        input_ids = []
        for i in range(len(doc_chunk_ids)):
            if i == 0:
                temp = doc_chunk_ids[i]
            else:
                # Align with K/V slicing that starts at s_start_1_len
                temp = doc_chunk_ids[i][s_start_1_len:]
            input_ids += temp
        return input_ids
