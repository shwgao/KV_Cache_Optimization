## How to use the full prefill compute in cache_blend
1. Please make sure you have installed the cache_blend package. **Don't** install the vllm package through `pip install vllm`.


2. Model initialization:
    ```python
    from vllm import LLM, SamplingParams

    eval_dataset = load_dataset("inputs/musique_s.json")

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.8,dtype=torch.float16, enforce_eager=True,)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    llm.set_tokenizer(tokenizer)
    ```

3. Get the handle of the `cache_fuse_metadata`:
    ```python
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    ```

4. Set the cache_fuse_metadata to False:
    ```python
    # You basically only need to set the cache_fuse_metadata['check'] to do the full prefill, the LLM will compute the the prefill from the scratch.
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False
    ```

5. Start the prefill compute:
    ```python
    output = llm.generate([input_prompt], sampling_params)
    ```

## How to prepare the KV Cache for chunks.
1. Get the KV cache for every chunk:
    ```python
    # set the collect to true that the LLM will save the KV cache into `self_attn.hack_kv`
    cache_fuse_metadata['collect'] = False
    chunk_prompts = ...

    llm.generate(chunk_prompts, sampling_params) 

    # get the KV cache for the model layer
    llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    cache_k = []
    cache_v = []
    for j in range(num_layer):
        cache_k.append(llm_layers[j].self_attn.hack_kv[0])
        cache_v.append(llm_layers[j].self_attn.hack_kv[1])
    ```

## Pipeline Logic Problems
1. Don't need `retriever` during prefill stage, because we need to compute full kv cache for all chunks.
    ```python
    # The part and the related should be removed.
    retriever = None
    if maybe_existing is None:
        rconf = RetrievalConfig(
            model_id=cfg["retrieval"]["model_id"],
            dataset_name=cfg["retrieval"]["dataset_name"],
            r_text_index_key=cfg["retrieval"]["r_text_index_key"],
            doc_key=cfg["retrieval"]["doc_key"],
            question_key=cfg["retrieval"]["question_key"],
            retrieved_key=cfg["retrieval"]["retrieved_key"],
            page_id_key=cfg["retrieval"]["page_id_key"],
            top_k=int(cfg["retrieval"]["top_k"]),
        )
        retriever = ColbertRetrieval(rconf)
    ```

2. Make sure we should use top-k chunks for prefill?

4. The fourth step speculative prediction is useless, as you put all decoding logic in the `scheduler.run`.

5. The `scheduler.run` is too heavy for every step, as you need to compute scores for all chunks and decide which chunks to promote or evict. You need to make sure the TOP is as good as regular decoding. I am thinking the 

6. If the speculative decoding is needed, then it should be combined within the `Scheduler`. However, I highly recommend you to reconstruct the `Scheduler`. It's better to split the model running, speculative predicting and KV cache scheduler to three different classes. Also multiprocessing is also recommended. 

7. Use async to prefetch and evict the KV cache.

8. For the `scheduler.run`, the logic of your implementation is not correct. No actual token generated for every step, which I am suppose you want to generate one token per step. 

9. Make sure only compute part of attention for the chunks is correct, as I didn't see the how to compute the output in your `scheduler.run`.