"""
Debug script to test transformers import and basic functionality with KV Cache
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path

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
        dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")
    
    print(f"✓ Mistral model loaded successfully")
        
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
        
        print(f"Input prompt: {input_prompt[:100]}...")
    
        with torch.no_grad():
            # Use KV Cache for step-by-step generation
            current_input = torch.tensor([input_ids], device="cuda")
            generated_tokens = []
            max_tokens = 20  # Maximum number of tokens to generate
            past_key_values = None  # Initialize KV Cache
            
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
                    print(f"Step {step+1}: End token encountered, stopping generation")
                    break
                
                generated_tokens.append(new_token)
                
                # Decode and display currently generated text
                current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"Step {step+1}: New token={new_token}, Current text='{current_text}'")
                
                # Display KV Cache information
                if past_key_values is not None:
                    cache_size = sum(kv[0].shape[2] for kv in past_key_values) // len(past_key_values)
                    print(f"  KV Cache length: {cache_size}")
            
            # Final results
            final_generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"\nFinal generated text: '{final_generated_text}'")
            
            # Complete text (input + generated)
            full_text = tokenizer.decode(input_ids + generated_tokens, skip_special_tokens=True)
            print(f"Complete text: {full_text}")
            
            # Display performance information
            print(f"\nPerformance statistics:")
            print(f"Generated token count: {len(generated_tokens)}")
            print(f"Input token count: {len(input_ids)}")
            print(f"Total token count: {len(input_ids) + len(generated_tokens)}")
    
    return True


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
        dtype=torch.float16,
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
    
    # print("\n=== KV Cache Optimized Generation Test ===")
    # Test Mistral with KV Cache optimization
    success2 = test_mistral_specific()
    
    print(f"\n=== Test Summary ===")
    print(f"Basic generation test: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"KV Cache test: {'✓ PASSED' if success2 else '✗ FAILED'}")
    
    print("\nDebug tests completed!")
