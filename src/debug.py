"""
Debug script to test transformers import and basic functionality with KV Cache

This script includes implementations of sparse attention mechanisms:

1. test_mistral_sparse_attention(): 
   - Prefill phase: Normal computation for all chunks
   - Decode phase: Fixed ratio of chunks (e.g., 60% of prefill chunks)

2. test_mistral_adaptive_sparse_attention():
   - Prefill phase: Normal computation for all chunks  
   - Decode phase: Dynamic chunk selection based on relevance to generated tokens
   - Re-evaluates chunk relevance every few steps during generation

3. compare_sparse_attention_methods():
   - Compares performance of different sparse attention approaches

Key Features:
- Memory efficiency: Only uses subset of prefill chunks during decode
- Dynamic selection: Adapts chunk usage based on generation context
- Performance monitoring: Tracks timing, memory usage, and throughput
- Relevance scoring: Uses word overlap to determine chunk relevance

Usage:
    python debug.py  # Runs all sparse attention tests and comparison
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path


DEBUG = False

# Load dataset with error handling
eval_dataset = load_dataset("inputs/musique_s.json")

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"


def test_mistral_specific():
    """Test Mistral model specifically with KV Cache optimization"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    ).to("cuda")
    
    print(f"✓ Mistral model loaded successfully")
    
    global DEBUG
    f1 = []
    ttft = []
    tpot = []
        
    run_idx = 0
    max_run = 5000
    for ex in eval_dataset:
        run_idx += 1
        if run_idx > max_run:
            break
        
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
        q_ids = tokenizer.encode(q_prompt)[1:]

        s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
        s_start_len = len(s_start_full) + 1

        #s_start = [518, 25580, 29962]
        s_start = []
        s_start_1_len = len(s_start) + 1

        #s_end = [518, 29914, 25580, 29962]
        s_end = [733, 28748, 16289, 28793]
        s_end_len = len(s_end)
        old_kvs = []

        doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
        doc_chunk_ids = [s_start_full] + doc_chunk_ids
        doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]
        
        last_len = len([q_ids+s_end])
        
        input_ids = []

        for i in range(len(doc_chunk_ids)):
            if i == 0:
                temp_ids = doc_chunk_ids[i]
            else:
                temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
            input_ids += temp_ids
            
        input_prompt = tokenizer.decode(input_ids)
        
        if DEBUG:
            print(f"Input prompt: {input_prompt[:100]}...")
    
        with torch.no_grad():
            # Use KV Cache for step-by-step generation
            current_input = torch.tensor([input_ids], device="cuda")
            generated_tokens = []
            max_tokens = 20  # Maximum number of tokens to generate
            past_key_values = None  # Initialize KV Cache
            
            if DEBUG:
                print("Starting KV Cache step-by-step generation:")
            
            for step in range(max_tokens):
                if step == 0:
                    # First time: process entire input sequence, generate KV Cache
                    outputs = model(
                        current_input,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
                    # Get the first generated token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                else:
                    # Subsequent steps: only process new token, use cached KV
                    outputs = model(
                        next_token.unsqueeze(-1),  # Only pass new token
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    # Update KV Cache
                    past_key_values = outputs.past_key_values
                    # Get next token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                
                new_token = next_token.item()
                
                # Check if end token is encountered
                if new_token == tokenizer.eos_token_id:
                    if DEBUG:
                        print(f"Step {step+1}: End token encountered, stopping generation")
                    break
                
                generated_tokens.append(new_token)
                
                # Decode and display currently generated text
                current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if DEBUG:
                    print(f"Step {step+1}: New token={new_token}, Current text='{current_text}'")
                
                # Display KV Cache information
                if past_key_values is not None:
                    cache_size = sum(kv[0].shape[2] for kv in past_key_values) // len(past_key_values)
                    if DEBUG:
                        print(f"  KV Cache length: {cache_size}")
            
            # Final results
            final_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if DEBUG:
                print(f"\nFinal generated text: '{final_generated_text}'")
                
            print(f"generated text: {final_generated_text}")
            print(f"answers: {answers}")
            
            f1.append(max(compute_f1(final_generated_text, answer, tokenizer) for answer in answers))
            print(f"F1: {f1[-1]:.3f}")
            print(f"average F1: {sum(f1)/len(f1):.3f}")
            
            # Display performance information
            if DEBUG:
                print(f"\nPerformance statistics:")
                print(f"Generated token count: {len(generated_tokens)}")
                print(f"Input token count: {len(input_ids)}")
                print(f"Total token count: {len(input_ids) + len(generated_tokens)}")
    
    print(f"------------>Average:<------------")
    print(f"F1: {sum(f1)/len(f1):.3f}")
      
    return True


def test_mistral_sparse_attention():
    """Test Mistral model with sparse attention: normal prefill, limited chunk decode using sparse KV cache."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    ).to("cuda")
    
    print(f"✓ Mistral sparse attention model loaded successfully")
    
    # Configuration for sparse attention
    max_prefill_chunks = 12  # Maximum chunks to use in prefill
    decode_chunk_ratio = 0.2  # Use 60% of prefill chunks during decode
    max_tokens = 20
    
    global DEBUG
    
    ttft = []
    f1 = []
    tpot = []
    
    
    run_idx = 0
    max_run = 5000
    for ex in eval_dataset:
        run_idx += 1
        if run_idx > max_run:
            break
        
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
        q_ids = tokenizer.encode(q_prompt)[1:]

        s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
        s_start_len = len(s_start_full) + 1

        s_start = []
        s_start_1_len = len(s_start) + 1

        s_end = [733, 28748, 16289, 28793]
        s_end_len = len(s_end)

        # Separate prefix, doc chunks, and query
        prefix_chunk = s_start_full  # prefix_prompt chunk
        doc_chunks = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]  # document chunks
        query_chunk = s_start+q_ids+s_end  # query_prompt chunk
        
        # Build full input sequence with proper chunk identification
        input_ids = []
        chunk_boundaries = []  # Track where each chunk starts/ends
        chunk_types = []  # Track chunk types: 'prefix', 'doc', 'query'
        
        # Add prefix chunk (always included)
        chunk_start = len(input_ids)
        input_ids += prefix_chunk
        chunk_end = len(input_ids)
        chunk_boundaries.append((chunk_start, chunk_end))
        chunk_types.append('prefix')
        
        # Add document chunks (can be selectively included)
        for i, doc_chunk in enumerate(doc_chunks):
            chunk_start = len(input_ids)
            # Remove overlap with previous chunk
            if i == 0:
                temp_ids = doc_chunk[s_start_1_len-1:]
            else:
                temp_ids = doc_chunk[s_start_1_len-1:]
            input_ids += temp_ids
            chunk_end = len(input_ids)
            chunk_boundaries.append((chunk_start, chunk_end))
            chunk_types.append('doc')
        
        # Add query chunk (always included)
        chunk_start = len(input_ids)
        query_temp_ids = query_chunk[s_start_1_len-1:]
        input_ids += query_temp_ids
        chunk_end = len(input_ids)
        chunk_boundaries.append((chunk_start, chunk_end))
        chunk_types.append('query')
        
        # Identify chunk indices by type
        prefix_chunk_idx = [i for i, t in enumerate(chunk_types) if t == 'prefix'][0]
        doc_chunk_indices = [i for i, t in enumerate(chunk_types) if t == 'doc']
        query_chunk_idx = [i for i, t in enumerate(chunk_types) if t == 'query'][0]
        
        if DEBUG:
            print(f"Prefix chunk index: {prefix_chunk_idx}")
            print(f"Document chunk indices: {doc_chunk_indices}")
            print(f"Query chunk index: {query_chunk_idx}")
                
            input_prompt = tokenizer.decode(input_ids)
            print(f"Input prompt: {input_prompt[:100]}...")
            print(f"Total chunks: {len(chunk_boundaries)}")
            print(f"Chunk types: {chunk_types}")
            print(f"Chunk boundaries: {chunk_boundaries}")
    
        with torch.no_grad():
            # PHASE 1: PREFILL - Use normal computation for all chunks
            if DEBUG:
                print("\n=== PREFILL PHASE ===")
            prefill_start_time = time.time()
            
            # For prefill, we always include prefix + selected doc chunks + query
            # Select document chunks for prefill (up to max_prefill_chunks)
            selected_doc_chunks = min(len(doc_chunk_indices), max_prefill_chunks - 2)  # -2 for prefix and query
            prefill_doc_indices = doc_chunk_indices[:selected_doc_chunks]
            
            # Build prefill sequence: prefix + selected docs + query
            prefill_chunk_indices = [prefix_chunk_idx] + prefill_doc_indices + [query_chunk_idx]
            prefill_input_ids = []
            
            for chunk_idx in prefill_chunk_indices:
                start, end = chunk_boundaries[chunk_idx]
                prefill_input_ids += input_ids[start:end]
            
            prefill_input = torch.tensor([prefill_input_ids], device="cuda")
            
            if DEBUG:
                print(f"Prefill using {len(prefill_chunk_indices)} chunks ({len(prefill_input_ids)} tokens)")
                print(f"Prefill chunk indices: {prefill_chunk_indices}")
                print(f"Prefill chunk types: {[chunk_types[i] for i in prefill_chunk_indices]}")
            
            # Normal forward pass for prefill
            prefill_outputs = model(
                prefill_input,
                use_cache=True,
                return_dict=True
            )
            past_key_values = prefill_outputs.past_key_values
            
            prefill_time = time.time() - prefill_start_time
            
            if DEBUG:
                print(f"Prefill completed in {prefill_time:.3f}s")
            
            # PHASE 2: DECODE - Use sparse attention with selected chunks
            if DEBUG:
                print("\n=== DECODE PHASE (SPARSE ATTENTION) ===")
            
            # For decode, we always include prefix + selected doc chunks + query
            # Select document chunks for decode (subset of prefill doc chunks)
            decode_doc_chunks = max(1, int(len(prefill_doc_indices) * decode_chunk_ratio))
            decode_doc_indices = prefill_doc_indices[:decode_doc_chunks]
            
            # Build decode sequence: prefix + selected docs + query
            decode_chunk_indices = [prefix_chunk_idx] + decode_doc_indices + [query_chunk_idx]
            
            if DEBUG:
                print(f"Using decode with {len(decode_chunk_indices)} chunks (indices: {decode_chunk_indices})")
                print(f"Decode chunk types: {[chunk_types[i] for i in decode_chunk_indices]}")
                print(f"Document chunks used: {len(decode_doc_indices)}/{len(doc_chunk_indices)}")
            
            # Create sparse KV cache by selecting only the decode chunks
            sparse_kv_cache = _create_sparse_kv_cache_from_chunks(
                past_key_values, 
                chunk_boundaries, 
                decode_chunk_indices,
                prefill_chunk_indices
            )
            
            generated_tokens = []
            decode_start_time = time.time()
            
            # Calculate the total sequence length from prefill
            prefill_seq_len = len(prefill_input_ids)
            
            for step in range(max_tokens):
                if step == 0:
                    # First decode step - use sparse KV cache
                    next_token_logits = prefill_outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                else:
                    # Subsequent steps - use sparse KV cache with correct position encoding
                    # Position ID should continue from the original prefill sequence length
                    current_position = prefill_seq_len + step - 1
                    position_ids = torch.tensor([[current_position]], device="cuda")
                    
                    outputs = model(
                        next_token.unsqueeze(-1),
                        past_key_values=sparse_kv_cache,
                        position_ids=position_ids,  # CRITICAL FIX: Provide correct position IDs
                        use_cache=True,
                        return_dict=True
                    )
                    sparse_kv_cache = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                
                new_token = next_token.item()
                
                # Check if end token is encountered
                if new_token == tokenizer.eos_token_id:
                    # print(f"Step {step+1}: End token encountered, stopping generation")
                    break
                
                generated_tokens.append(new_token)
                
                # Decode and display currently generated text
                current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # print(f"Step {step+1}: New token={new_token}, Current text='{current_text}'")
                
                # Display sparse attention information
                if sparse_kv_cache is not None:
                    cache_size = sum(kv[0].shape[2] for kv in sparse_kv_cache) // len(sparse_kv_cache)
                    # print(f"  Sparse KV Cache length: {cache_size}")
            
            
            decode_time = time.time() - decode_start_time
            
            # Final results
            final_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"\nFinal generated text: '{final_generated_text}'")
            print(f"answers: {answers}")
            
            f1.append(max(compute_f1(final_generated_text, answer, tokenizer) for answer in answers))
            ttft.append(prefill_time)
            tpot.append(decode_time/len(generated_tokens))
            
            print(f"TTFT: {ttft[-1]:.3f}s")
            print(f"TPOT: {tpot[-1]:.3f}s/token")
            print(f"F1: {f1[-1]:.3f}")
            
    print(f"------------>Average:<------------")
    print(f"TTFT: {sum(ttft)/len(ttft):.3f}s")
    print(f"TPOT: {sum(tpot)/len(tpot):.3f}s/token")
    print(f"F1: {sum(f1)/len(f1):.3f}")
    
            
            # # Performance statistics
            # print(f"\n=== SPARSE ATTENTION PERFORMANCE ===")
            # print(f"Prefill phase:")
            # print(f"  - Total chunks used: {len(prefill_chunk_indices)}/{len(chunk_boundaries)}")
            # print(f"  - Document chunks used: {len(prefill_doc_indices)}/{len(doc_chunk_indices)}")
            # print(f"  - Tokens processed: {len(prefill_input_ids)}")
            # print(f"  - Time: {prefill_time:.3f}s")
            # print(f"Decode phase:")
            # print(f"  - Active chunks used: {len(decode_chunk_indices)}")
            # print(f"  - Document chunks used: {len(decode_doc_indices)}/{len(doc_chunk_indices)}")
            # print(f"  - Generated tokens: {len(generated_tokens)}")
            # print(f"  - Time: {decode_time:.3f}s")
            # print(f"  - TPOT: {decode_time/len(generated_tokens):.3f}s/token")
            # print(f"Overall:")
            # print(f"  - Total time: {prefill_time + decode_time:.3f}s")
            # print(f"  - Document chunk efficiency: {len(decode_doc_indices)}/{len(doc_chunk_indices)} ({len(decode_doc_indices)/len(doc_chunk_indices):.1%})")
            # print(f"  - Memory efficiency: Using sparse KV cache with selected chunks only")
    
    return True


def _create_sparse_kv_cache_from_chunks(full_kv_cache, chunk_boundaries, decode_chunk_indices, prefill_chunk_indices):
    """
    Create a sparse KV cache by selecting only specific chunks from the prefill cache.
    
    IMPORTANT: This function maintains the original position encoding by keeping the 
    original token positions in the KV cache, ensuring attention calculations remain correct.
    
    Args:
        full_kv_cache: Complete KV cache from prefill
        chunk_boundaries: List of (start, end) positions for each chunk
        decode_chunk_indices: Indices of chunks to keep in decode
        prefill_chunk_indices: Indices of chunks used in prefill
    
    Returns:
        Sparse KV cache containing only selected chunks with preserved position encoding
    """
    if not decode_chunk_indices:
        return full_kv_cache
    
    # Calculate which token positions to keep based on chunk boundaries
    keep_positions = []
    for decode_chunk_idx in decode_chunk_indices:
        if decode_chunk_idx < len(chunk_boundaries):
            start, end = chunk_boundaries[decode_chunk_idx]
            keep_positions.extend(range(start, end))
    
    if not keep_positions:
        return full_kv_cache
    
    # Sort positions to maintain order
    keep_positions = sorted(keep_positions)
    
    # Create sparse KV cache by selecting only the keep positions
    # CRITICAL: We keep the original positions to preserve RoPE encoding
    sparse_kv_cache = []
    for layer_kv in full_kv_cache:
        keys, values = layer_kv
        # Select only the positions we want to keep
        sparse_keys = keys[:, :, keep_positions, :]
        sparse_values = values[:, :, keep_positions, :]
        sparse_kv_cache.append((sparse_keys, sparse_values))
    
    # Return the same type as the original cache to maintain compatibility
    if hasattr(full_kv_cache, '__class__'):
        # Try to create the same type of object
        try:
            return full_kv_cache.__class__(sparse_kv_cache)
        except:
            # Fallback to tuple if we can't recreate the original type
            return tuple(sparse_kv_cache)
    else:
        return tuple(sparse_kv_cache)


def _create_sparse_kv_cache(full_kv_cache, chunk_boundaries, selected_chunk_indices):
    """
    Create a sparse KV cache by selecting only specific chunks from the full cache.
    
    Args:
        full_kv_cache: Complete KV cache from prefill
        chunk_boundaries: List of (start, end) positions for each chunk
        selected_chunk_indices: Indices of chunks to keep in sparse cache
    
    Returns:
        Sparse KV cache containing only selected chunks
    """
    if not selected_chunk_indices:
        return full_kv_cache
    
    # Calculate which token positions to keep
    keep_positions = []
    for chunk_idx in selected_chunk_indices:
        if chunk_idx < len(chunk_boundaries):
            start, end = chunk_boundaries[chunk_idx]
            keep_positions.extend(range(start, end))
    
    if not keep_positions:
        return full_kv_cache
    
    # Create sparse KV cache by selecting only the keep positions
    # We need to maintain the same structure as the original cache
    sparse_kv_cache = []
    for layer_kv in full_kv_cache:
        keys, values = layer_kv
        # Select only the positions we want to keep
        sparse_keys = keys[:, :, keep_positions, :]
        sparse_values = values[:, :, keep_positions, :]
        sparse_kv_cache.append((sparse_keys, sparse_values))
    
    # Return the same type as the original cache to maintain compatibility
    if hasattr(full_kv_cache, '__class__'):
        # Try to create the same type of object
        try:
            return full_kv_cache.__class__(sparse_kv_cache)
        except:
            # Fallback to tuple if we can't recreate the original type
            return tuple(sparse_kv_cache)
    else:
        return tuple(sparse_kv_cache)


def test_mistral_adaptive_sparse_attention():
    """Test Mistral model with adaptive sparse attention: dynamic chunk selection based on relevance using sparse KV cache."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    ).to("cuda")
    
    print(f"✓ Mistral adaptive sparse attention model loaded successfully")
    
    # Configuration for adaptive sparse attention
    max_prefill_chunks = 8  # Maximum chunks to use in prefill
    min_decode_chunks = 2   # Minimum chunks to keep during decode
    max_decode_chunks = 5   # Maximum chunks to keep during decode
    relevance_threshold = 0.3  # Threshold for chunk relevance
    max_tokens = 20
    
    run_idx = 0
    max_run = 1
    for ex in eval_dataset:
        run_idx += 1
        if run_idx > max_run:
            break
        
        answers = ex["answers"]
        doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
        doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
        q_ids = tokenizer.encode(q_prompt)[1:]

        s_start_full = [733, 16289, 28793] + tokenizer.encode(prefix_prompt)[1:]
        s_start_len = len(s_start_full) + 1

        s_start = []
        s_start_1_len = len(s_start) + 1

        s_end = [733, 28748, 16289, 28793]
        s_end_len = len(s_end)

        # Separate prefix, doc chunks, and query
        prefix_chunk = s_start_full  # prefix_prompt chunk
        doc_chunks = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]  # document chunks
        query_chunk = s_start+q_ids+s_end  # query_prompt chunk
        
        # Build full input sequence with proper chunk identification
        input_ids = []
        chunk_boundaries = []  # Track where each chunk starts/ends
        chunk_types = []  # Track chunk types: 'prefix', 'doc', 'query'
        chunk_texts = []  # Store original text for relevance calculation
        
        # Add prefix chunk (always included)
        chunk_start = len(input_ids)
        input_ids += prefix_chunk
        chunk_end = len(input_ids)
        chunk_boundaries.append((chunk_start, chunk_end))
        chunk_types.append('prefix')
        chunk_texts.append(tokenizer.decode(prefix_chunk))
        
        # Add document chunks (can be selectively included)
        for i, doc_chunk in enumerate(doc_chunks):
            chunk_start = len(input_ids)
            # Remove overlap with previous chunk
            if i == 0:
                temp_ids = doc_chunk[s_start_1_len-1:]
            else:
                temp_ids = doc_chunk[s_start_1_len-1:]
            input_ids += temp_ids
            chunk_end = len(input_ids)
            chunk_boundaries.append((chunk_start, chunk_end))
            chunk_types.append('doc')
            chunk_texts.append(tokenizer.decode(temp_ids))
        
        # Add query chunk (always included)
        chunk_start = len(input_ids)
        query_temp_ids = query_chunk[s_start_1_len-1:]
        input_ids += query_temp_ids
        chunk_end = len(input_ids)
        chunk_boundaries.append((chunk_start, chunk_end))
        chunk_types.append('query')
        chunk_texts.append(tokenizer.decode(query_temp_ids))
        
        # Identify chunk indices by type
        prefix_chunk_idx = [i for i, t in enumerate(chunk_types) if t == 'prefix'][0]
        doc_chunk_indices = [i for i, t in enumerate(chunk_types) if t == 'doc']
        query_chunk_idx = [i for i, t in enumerate(chunk_types) if t == 'query'][0]
            
        input_prompt = tokenizer.decode(input_ids)
        print(f"Input prompt: {input_prompt[:100]}...")
        print(f"Total chunks: {len(chunk_boundaries)}")
    
        with torch.no_grad():
            # PHASE 1: PREFILL - Use normal computation for all chunks
            print("\n=== PREFILL PHASE ===")
            prefill_start_time = time.time()
            
            # For prefill, we always include prefix + selected doc chunks + query
            # Select document chunks for prefill (up to max_prefill_chunks)
            selected_doc_chunks = min(len(doc_chunk_indices), max_prefill_chunks - 2)  # -2 for prefix and query
            prefill_doc_indices = doc_chunk_indices[:selected_doc_chunks]
            
            # Build prefill sequence: prefix + selected docs + query
            prefill_chunk_indices = [prefix_chunk_idx] + prefill_doc_indices + [query_chunk_idx]
            prefill_input_ids = []
            
            for chunk_idx in prefill_chunk_indices:
                start, end = chunk_boundaries[chunk_idx]
                prefill_input_ids += input_ids[start:end]
            
            prefill_input = torch.tensor([prefill_input_ids], device="cuda")
            
            print(f"Prefill using {len(prefill_chunk_indices)} chunks ({len(prefill_input_ids)} tokens)")
            print(f"Prefill chunk indices: {prefill_chunk_indices}")
            print(f"Prefill chunk types: {[chunk_types[i] for i in prefill_chunk_indices]}")
            
            # Normal forward pass for prefill
            prefill_outputs = model(
                prefill_input,
                use_cache=True,
                return_dict=True
            )
            past_key_values = prefill_outputs.past_key_values
            
            prefill_time = time.time() - prefill_start_time
            print(f"Prefill completed in {prefill_time:.3f}s")
            
            # PHASE 2: ADAPTIVE DECODE - Use dynamic chunk selection with sparse KV cache
            print("\n=== ADAPTIVE DECODE PHASE ===")
            
            generated_tokens = []
            decode_start_time = time.time()
            
            # Calculate the total sequence length from prefill
            prefill_seq_len = len(prefill_input_ids)
            
            # For decode, we always include prefix + selected doc chunks + query
            # Start with subset of prefill doc chunks
            initial_doc_chunks = min(len(prefill_doc_indices), max_decode_chunks - 2)  # -2 for prefix and query
            current_doc_indices = prefill_doc_indices[:initial_doc_chunks]
            current_chunk_indices = [prefix_chunk_idx] + current_doc_indices + [query_chunk_idx]
            
            print(f"Initial decode chunks: {current_chunk_indices}")
            print(f"Initial decode chunk types: {[chunk_types[i] for i in current_chunk_indices]}")
            print(f"Document chunks used: {len(current_doc_indices)}/{len(doc_chunk_indices)}")
            
            # Create initial sparse KV cache
            sparse_kv_cache = _create_sparse_kv_cache_from_chunks(
                past_key_values, 
                chunk_boundaries, 
                current_chunk_indices,
                prefill_chunk_indices
            )
            
            for step in range(max_tokens):
                if step == 0:
                    # First decode step
                    next_token_logits = prefill_outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                else:
                    # Adaptive chunk selection every few steps
                    if step % 3 == 0 and step > 0:  # Re-evaluate chunks every 3 steps
                        # Only select from document chunks, always keep prefix and query
                        doc_chunk_texts = [chunk_texts[i] for i in doc_chunk_indices]
                        selected_doc_indices = _select_relevant_chunks(
                            doc_chunk_texts,
                            generated_tokens,
                            tokenizer,
                            min_decode_chunks - 2,  # -2 for prefix and query
                            max_decode_chunks - 2,  # -2 for prefix and query
                            relevance_threshold
                        )
                        
                        # Map back to original indices
                        current_doc_indices = [doc_chunk_indices[i] for i in selected_doc_indices]
                        current_chunk_indices = [prefix_chunk_idx] + current_doc_indices + [query_chunk_idx]
                        print(f"Step {step}: Updated chunk selection: {current_chunk_indices}")
                        print(f"Step {step}: Updated chunk types: {[chunk_types[i] for i in current_chunk_indices]}")
                        
                        # Update sparse KV cache with new chunk selection
                        sparse_kv_cache = _create_sparse_kv_cache_from_chunks(
                            past_key_values, 
                            chunk_boundaries, 
                            current_chunk_indices,
                            prefill_chunk_indices
                        )
                    
                    # Continue generation with sparse KV cache and correct position encoding
                    # Position ID should continue from the original prefill sequence length
                    current_position = prefill_seq_len + step - 1
                    position_ids = torch.tensor([[current_position]], device="cuda")
                    
                    outputs = model(
                        next_token.unsqueeze(-1),
                        past_key_values=sparse_kv_cache,
                        position_ids=position_ids,  # CRITICAL FIX: Provide correct position IDs
                        use_cache=True,
                        return_dict=True
                    )
                    sparse_kv_cache = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / 0.7, dim=-1), 
                        num_samples=1
                    ).squeeze(-1)
                
                new_token = next_token.item()
                
                # Check if end token is encountered
                if new_token == tokenizer.eos_token_id:
                    print(f"Step {step+1}: End token encountered, stopping generation")
                    break
                
                generated_tokens.append(new_token)
                
                # Decode and display currently generated text
                current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"Step {step+1}: New token={new_token}, Current text='{current_text}'")
                print(f"  Active chunks: {current_chunk_indices}")
            
            decode_time = time.time() - decode_start_time
            
            # Final results
            final_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"\nFinal generated text: '{final_generated_text}'")
            
            # Performance statistics
            print(f"\n=== ADAPTIVE SPARSE ATTENTION PERFORMANCE ===")
            print(f"Prefill phase:")
            print(f"  - Total chunks used: {len(prefill_chunk_indices)}/{len(chunk_boundaries)}")
            print(f"  - Document chunks used: {len(prefill_doc_indices)}/{len(doc_chunk_indices)}")
            print(f"  - Tokens processed: {len(prefill_input_ids)}")
            print(f"  - Time: {prefill_time:.3f}s")
            print(f"Decode phase:")
            print(f"  - Final active chunks: {len(current_chunk_indices)}")
            print(f"  - Final document chunks used: {len(current_doc_indices)}/{len(doc_chunk_indices)}")
            print(f"  - Generated tokens: {len(generated_tokens)}")
            print(f"  - Time: {decode_time:.3f}s")
            print(f"  - TPOT: {decode_time/len(generated_tokens):.3f}s/token")
            print(f"Overall:")
            print(f"  - Total time: {prefill_time + decode_time:.3f}s")
            print(f"  - Document chunk efficiency: {len(current_doc_indices)}/{len(doc_chunk_indices)} ({len(current_doc_indices)/len(doc_chunk_indices):.1%})")
            print(f"  - Memory efficiency: Using sparse KV cache with selected chunks only")
    
    return True


def _select_relevant_chunks(chunk_texts, generated_tokens, tokenizer, min_chunks, max_chunks, threshold):
    """
    Select most relevant chunks based on generated tokens and chunk content.
    
    Args:
        chunk_texts: List of chunk text content
        generated_tokens: List of generated token IDs
        tokenizer: Tokenizer for text processing
        min_chunks: Minimum number of chunks to keep
        max_chunks: Maximum number of chunks to keep
        threshold: Relevance threshold for chunk selection
    
    Returns:
        List of chunk indices to keep
    """
    if not generated_tokens or len(chunk_texts) <= min_chunks:
        return list(range(min(len(chunk_texts), max_chunks)))
    
    # Convert generated tokens to text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).lower()
    
    # Calculate relevance scores for each chunk
    relevance_scores = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_text_lower = chunk_text.lower()
        
        # Simple relevance scoring based on word overlap
        generated_words = set(generated_text.split())
        chunk_words = set(chunk_text_lower.split())
        
        if len(generated_words) == 0:
            relevance_score = 0.0
        else:
            overlap = len(generated_words.intersection(chunk_words))
            relevance_score = overlap / len(generated_words)
        
        relevance_scores.append((i, relevance_score))
    
    # Sort by relevance score
    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select chunks above threshold, but ensure we have at least min_chunks
    selected_indices = []
    for idx, score in relevance_scores:
        if len(selected_indices) >= max_chunks:
            break
        if score >= threshold or len(selected_indices) < min_chunks:
            selected_indices.append(idx)
    
    # Sort indices to maintain order
    selected_indices.sort()
    
    return selected_indices


def test_basic_generation():
    """Test basic generation without KV Cache for comparison"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True
    ).to("cuda")
    
    print(f"✓ Basic generation test - model loaded")
    
    # Simple test input
    test_input = "Hello, how are you today?"
    input_ids = tokenizer.encode(test_input, return_tensors="pt").to("cuda")
    
    print(f"Test input: {test_input}")
    
    with torch.no_grad():
        # Basic generation without KV Cache optimization
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True  # Still use cache but in standard way
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    return True


if __name__ == "__main__":
    print("Starting transformers debug tests with KV Cache...\n")
    
    # # Test basic generation
    # print("=== Basic Generation Test ===")
    # success1 = test_basic_generation()
    
    # Test Mistral with KV Cache optimization
    # success2 = test_mistral_specific()
    
    # Test sparse attention
    # print("\n=== Sparse Attention Test ===")
    success3 = test_mistral_sparse_attention()
    
    # # Test adaptive sparse attention
    # print("\n=== Adaptive Sparse Attention Test ===")
    # success4 = test_mistral_adaptive_sparse_attention()
    
    print("\nDebug tests completed!")
