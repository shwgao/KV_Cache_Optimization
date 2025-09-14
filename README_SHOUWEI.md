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
