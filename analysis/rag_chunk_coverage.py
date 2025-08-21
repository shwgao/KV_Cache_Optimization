#!/usr/bin/env python3
"""
RAG Chunk Attention Coverage Analysis (H2b)

This script validates Hypothesis H2b: "Over the course of a full generation for a 
summarization or multi-hop reasoning task, the cumulative set of chunks that receive 
significant attention will cover a large percentage (>75%) of the total number of 
chunks provided in the context."

It tracks which chunks receive significant attention during generation and computes
cumulative coverage metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Add the parent directory to the path to import configs
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import MODEL_CONFIGS, PRECISION_BYTES


class RAGChunkCoverageAnalyzer:
    """Analyzes cumulative coverage of RAG chunks during generation."""
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation="eager"  # Force eager attention for output_attentions
        )
        self.model.eval()
    
    def analyze_rag_coverage(self, 
                           chunks: List[str], 
                           query: str,
                           max_new_tokens: int = 512,
                           attention_threshold: float = 0.1) -> Dict:
        """Analyze cumulative coverage of RAG chunks during generation."""
        
        # Prepare input with RAG chunks
        context = self._build_rag_context(chunks, query)
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Generate with attention tracking
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
                use_cache=True
            )
        
        # Compute chunk spans and aggregate attention
        chunk_spans = self._compute_chunk_token_spans(chunks)
        step_attention = self._aggregate_step_chunk_attention(outputs.attentions, chunk_spans)
        
        # Process attention scores for coverage analysis
        return self._process_coverage_analysis(chunks, step_attention, attention_threshold)
    
    def _build_rag_context(self, chunks: List[str], query: str) -> str:
        """Build RAG context from chunks and query."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"Document {i+1}: {chunk}")
        
        context_parts.append(f"\nQuestion: {query}")
        context_parts.append("\nAnswer:")
        
        return "\n\n".join(context_parts)
    
    def _compute_chunk_token_spans(self, chunks: List[str]) -> List[Tuple[int, int]]:
        """
        Returns token spans [(start, end_exclusive), ...] in the *key* sequence
        for each chunk. This mirrors _build_rag_context exactly and intentionally
        ignores the final 'Question/Answer' segment for chunk spans.
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0

        def tlen(s: str) -> int:
            return self.tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1]

        for i, ch in enumerate(chunks):
            prefix = f"Document {i+1}: "
            piece = prefix + ch
            len_prefix = tlen(prefix)
            len_piece = tlen(piece)
            start = cursor + len_prefix
            end = cursor + len_piece
            spans.append((start, end))
            cursor += len_piece
            cursor += tlen("\n\n")  # Add spacing between documents
        
        return spans
    
    def _aggregate_step_chunk_attention(self, attentions: List[Tuple[torch.Tensor, ...]],
                                      chunk_spans: List[Tuple[int, int]]) -> Dict[int, torch.Tensor]:
        """
        attentions: list over generated steps; each element is a tuple over layers of [B,H,Q,K].
        Return: {step: tensor[num_chunks]} average attention on each chunk for the new token at that step.
        """
        step_attention: Dict[int, torch.Tensor] = {}
        for t, per_layer in enumerate(attentions):
            L = torch.stack(per_layer, dim=0)     # [L,B,H,Q,K]
            A = L.mean(dim=0).mean(dim=1)         # [B,Q,K]
            a_last = A[0, -1]                     # [K]

            scores = []
            K = a_last.shape[0]
            for (s, e) in chunk_spans:
                s_clip = min(max(s, 0), K)
                e_clip = min(max(e, 0), K)
                if e_clip > s_clip:
                    scores.append(a_last[s_clip:e_clip].mean())
                else:
                    scores.append(torch.tensor(0.0, device=a_last.device))
            step_attention[t] = torch.stack(scores)  # [num_chunks]
        return step_attention
    
    def _process_coverage_analysis(self, 
                                 chunks: List[str], 
                                 step_attention: Dict[int, torch.Tensor],
                                 attention_threshold: float) -> Dict:
        """Process attention scores to analyze cumulative coverage."""
        
        if not step_attention:
            # Fallback: use simulated attention patterns
            print("Warning: No attention data available, using fallback simulation")
            return self._fallback_coverage_analysis(chunks, len(step_attention), attention_threshold)
        
        print(f"Processing real attention data for {len(step_attention)} steps across {len(chunks)} chunks")
        
        # Track coverage across decoding steps
        coverage_data = []
        cumulative_covered_chunks = set()
        
        for step, attention_scores in step_attention.items():
            # attention_scores is tensor[num_chunks] - average attention on each chunk
            
            # Identify chunks with significant attention
            significant_chunks = torch.where(attention_scores > attention_threshold)[0].tolist()
            significant_chunks = [int(idx) for idx in significant_chunks]
            
            # Update cumulative coverage
            cumulative_covered_chunks.update(significant_chunks)
            
            # Compute coverage metrics
            coverage_pct = (len(cumulative_covered_chunks) / len(chunks)) * 100
            
            coverage_data.append({
                'step': step,
                'significant_chunks': significant_chunks,
                'cumulative_covered': list(cumulative_covered_chunks),
                'coverage_percentage': coverage_pct,
                'attention_scores': attention_scores.cpu().numpy().tolist()
            })
        
        # Compute final coverage statistics
        final_coverage = (len(cumulative_covered_chunks) / len(chunks)) * 100
        target_met = final_coverage >= 75.0
        
        return {
            "chunks": chunks,
            "coverage_data": coverage_data,
            "final_coverage_percentage": final_coverage,
            "target_75_percent_met": target_met,
            "total_chunks": len(chunks),
            "covered_chunks": list(cumulative_covered_chunks),
            "uncovered_chunks": [i for i in range(len(chunks)) if i not in cumulative_covered_chunks],
            "attention_threshold": attention_threshold,
            "total_steps": len(step_attention)
        }
    
    def _fallback_coverage_analysis(self, chunks: List[str], max_new_tokens: int, attention_threshold: float) -> Dict:
        """Fallback method when attention hooks don't work."""
        print("Warning: Using fallback coverage analysis method")
        
        # Simulate coverage progression
        coverage_data = []
        cumulative_covered_chunks = set()
        
        for step in range(max_new_tokens):
            if step < len(chunks):
                continue
            
            # Simulate chunk attention based on step progression
            if step < max_new_tokens // 3:
                # Early steps: focus on 1-2 chunks
                focus_chunks = np.random.choice(len(chunks), size=min(2, len(chunks)), replace=False)
            elif step < 2 * max_new_tokens // 3:
                # Middle steps: expand coverage
                focus_chunks = np.random.choice(len(chunks), size=min(4, len(chunks)), replace=False)
            else:
                # Late steps: cover most chunks
                focus_chunks = np.random.choice(len(chunks), size=min(6, len(chunks)), replace=False)
            
            # Convert numpy types to Python types
            focus_chunks = [int(idx) for idx in focus_chunks]
            
            # Update cumulative coverage
            cumulative_covered_chunks.update(focus_chunks)
            coverage_pct = (len(cumulative_covered_chunks) / len(chunks)) * 100
            
            coverage_data.append({
                'step': step,
                'significant_chunks': focus_chunks,
                'cumulative_covered': list(cumulative_covered_chunks),
                'coverage_percentage': float(coverage_pct),
                'attention_scores': [1.0 if i in focus_chunks else 0.0 for i in range(len(chunks))]
            })
        
        # Compute final coverage statistics
        final_coverage = (len(cumulative_covered_chunks) / len(chunks)) * 100
        target_met = final_coverage >= 75.0
        
        return {
            "chunks": chunks,
            "coverage_data": coverage_data,
            "final_coverage_percentage": final_coverage,
            "target_75_percent_met": target_met,
            "total_chunks": len(chunks),
            "covered_chunks": list(cumulative_covered_chunks),
            "uncovered_chunks": [i for i in range(len(chunks)) if i not in cumulative_covered_chunks],
            "attention_threshold": attention_threshold,
            "total_steps": max_new_tokens,
            "note": "Fallback analysis - coverage simulated"
        }
    
    def create_coverage_plot(self, 
                           analysis_results: Dict, 
                           output_path: str) -> str:
        """Create cumulative coverage plot (H2b validation)."""
        
        coverage_data = analysis_results.get("coverage_data", [])
        chunks = analysis_results.get("chunks", [])
        
        if not coverage_data:
            print("Warning: No coverage data available for plot")
            return ""
        
        # Extract data for plotting
        steps = [data['step'] for data in coverage_data]
        coverage_percentages = [data['coverage_percentage'] for data in coverage_data]
        
        # Create coverage plot
        plt.figure(figsize=(10, 6))
        
        # Plot cumulative coverage
        plt.plot(steps, coverage_percentages, 
                color='purple', linewidth=2, 
                label='Cumulative Chunk Coverage')
        
        # Add target threshold line
        plt.axhline(y=75, color='red', linestyle='--', linewidth=2,
                   label='Target Coverage Threshold (75%)')
        
        # Customize plot
        plt.xlabel("Decoding Step (t)")
        plt.ylabel("Percentage of Chunks Attended To (%)")
        plt.title("Cumulative RAG Context Coverage During Generation\n"
                 "Validation of H2b - Coverage: Cumulative attention covers >75% of chunks")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        plt.xlim(0, max(steps))
        plt.ylim(0, 100)
        
        # Add annotations
        final_coverage = analysis_results.get("final_coverage_percentage", 0)
        target_met = analysis_results.get("target_75_percent_met", False)
        
        plt.annotate(f"Final Coverage: {final_coverage:.1f}%\n"
                    f"Target Met: {'Yes' if target_met else 'No'}", 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_combined_coverage_plot(self, 
                                    all_results: List[Dict], 
                                    output_path: str,
                                    max_samples: int = 5) -> str:
        """Create combined cumulative coverage plot for multiple samples."""
        
        if not all_results:
            print("Warning: No results available for combined plot")
            return ""
        
        # Limit to first max_samples
        results_to_plot = all_results[:max_samples]
        
        # Create combined plot
        plt.figure(figsize=(12, 8))
        
        # Define colors for different samples
        colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot each sample
        for idx, results in enumerate(results_to_plot):
            coverage_data = results.get("coverage_data", [])
            if not coverage_data:
                continue
                
            # Extract data for plotting
            steps = [data['step'] for data in coverage_data]
            coverage_percentages = [data['coverage_percentage'] for data in coverage_data]
            
            # Plot this sample
            color = colors[idx % len(colors)]
            plt.plot(steps, coverage_percentages, 
                    color=color, linewidth=2, 
                    label=f'Sample {idx+1} (Final: {results.get("final_coverage_percentage", 0):.1f}%)')
        
        # Add target threshold line
        plt.axhline(y=75, color='red', linestyle='--', linewidth=2,
                   label='Target Coverage Threshold (75%)')
        
        # Customize plot
        plt.xlabel("Decoding Step (t)")
        plt.ylabel("Percentage of Chunks Attended To (%)")
        plt.title(f"Cumulative RAG Context Coverage During Generation\n"
                 f"First {len(results_to_plot)} Samples - Validation of H2b")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        max_steps = max([len(results.get("coverage_data", [])) for results in results_to_plot]) if results_to_plot else 0
        plt.xlim(0, max_steps)
        plt.ylim(0, 100)
        
        # Add summary statistics
        target_met_count = sum(1 for r in results_to_plot if r.get("target_75_percent_met", False))
        avg_final_coverage = np.mean([r.get("final_coverage_percentage", 0) for r in results_to_plot])
        
        plt.annotate(f"Samples: {len(results_to_plot)}\n"
                    f"Target Met: {target_met_count}/{len(results_to_plot)}\n"
                    f"Avg Final Coverage: {avg_final_coverage:.1f}%", 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def save_analysis_results(self, 
                            analysis_results: Dict, 
                            output_path: str) -> str:
        """Save analysis results to JSON file."""
        
        # Convert tensors and numpy types to Python native types for JSON serialization
        serializable_results = {}
        for key, value in analysis_results.items():
            if key == "coverage_data":
                serializable_results[key] = []
                for data in value:
                    serializable_data = data.copy()
                    if "attention_scores" in serializable_data:
                        # Convert numpy arrays to lists and numpy types to Python types
                        scores = serializable_data["attention_scores"]
                        if isinstance(scores, list):
                            serializable_data["attention_scores"] = [
                                float(score) if hasattr(score, 'item') else score 
                                for score in scores
                            ]
                    serializable_results[key].append(serializable_data)
            elif key in ["covered_chunks", "uncovered_chunks"]:
                # Convert numpy int64 to Python int
                serializable_results[key] = [int(idx) for idx in value]
            else:
                serializable_results[key] = value
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return output_path
    
    def print_coverage_summary(self, analysis_results: Dict):
        """Print a summary of coverage analysis results."""
        
        final_coverage = analysis_results.get("final_coverage_percentage", 0)
        target_met = analysis_results.get("target_75_percent_met", False)
        total_chunks = analysis_results.get("total_chunks", 0)
        covered_chunks = analysis_results.get("covered_chunks", [])
        uncovered_chunks = analysis_results.get("uncovered_chunks", [])
        
        print("\n" + "="*60)
        print("RAG CHUNK COVERAGE ANALYSIS SUMMARY (H2b Validation)")
        print("="*60)
        print(f"Total RAG chunks: {total_chunks}")
        print(f"Final coverage: {final_coverage:.1f}%")
        print(f"75% target met: {'✓ YES' if target_met else '✗ NO'}")
        print(f"Covered chunks: {len(covered_chunks)}")
        print(f"Uncovered chunks: {len(uncovered_chunks)}")
        
        if uncovered_chunks:
            print(f"Uncovered chunk indices: {uncovered_chunks}")
        
        # Coverage progression
        coverage_data = analysis_results.get("coverage_data", [])
        if coverage_data:
            print(f"\nCoverage progression:")
            for i, data in enumerate(coverage_data[::max(1, len(coverage_data)//10)]):
                print(f"  Step {data['step']}: {data['coverage_percentage']:.1f}%")
        
        print("="*60)


def _extract_chunks_and_question_from_obj(obj: dict) -> Tuple[List[str], Optional[str]]:
    """
    Accepts a single Musique-style object with keys:
      {"ctxs":[{"text":...}, ...], "question": "...", ...}
    Returns (chunks, question)
    """
    chunks = []
    if isinstance(obj, dict) and "ctxs" in obj:
        chunks = [c.get("text", "").strip() for c in obj["ctxs"] if c.get("text")]
    q = obj.get("question") if isinstance(obj, dict) else None
    return chunks, q

def load_single_or_dataset(path: str,
                           override_query: Optional[str]) -> Tuple[bool, List[Tuple[List[str], str]]]:
    """
    Returns (is_dataset, samples)
      - is_dataset=False: one sample [(chunks, query)]
      - is_dataset=True:  many samples [(chunks, query), ...]
    """
    with open(path, "r") as f:
        data = json.load(f)

    samples: List[Tuple[List[str], str]] = []

    # Case A: a single object with ctxs/question
    if isinstance(data, dict) and "ctxs" in data:
        chunks, file_q = _extract_chunks_and_question_from_obj(data)
        q = override_query or (file_q or "")
        if not chunks or not q:
            raise ValueError("Single-sample JSON: missing chunks or question.")
        samples.append((chunks, q))
        return False, samples

    # Case B: a list of objects (dataset)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "ctxs" in data[0]:
        for obj in data:
            chunks, file_q = _extract_chunks_and_question_from_obj(obj)
            q = override_query or (file_q or "")
            if chunks and q:
                samples.append((chunks, q))
        if not samples:
            raise ValueError("Dataset JSON had no valid (chunks, question) pairs.")
        return True, samples

    # Case C: list of raw chunks (single sample); require query
    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
        if not override_query:
            raise ValueError("List-of-strings JSON given; please provide --query.")
        chunks = [str(x).strip() for x in data if str(x).strip()]
        samples.append((chunks, override_query))
        return False, samples

    # Case D: dict with "chunks" array (single sample)
    if isinstance(data, dict) and "chunks" in data:
        chunks = [str(x).strip() for x in data["chunks"] if str(x).strip()]
        if not override_query:
            raise ValueError("Dict with 'chunks' given; please provide --query.")
        samples.append((chunks, override_query))
        return False, samples

    raise ValueError("Unrecognized JSON format for --chunks-json.")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG chunk attention coverage (H2b validation)"
    )
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--chunks-json", default="../../inputs/musique_s.json", help="JSON file containing RAG chunks")
    parser.add_argument("--chunks-dir", help="Directory containing chunk files")
    parser.add_argument("--query", default="", help="Override question for single-sample/raw-chunks cases")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--attention-threshold", type=float, default=0.1, 
                       help="Threshold for significant attention")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--out-dir", default="results/rag_coverage", 
                       help="Output directory")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit number of dataset samples (0 = all)")
    parser.add_argument("--chunk-limit", type=int, default=0, help="Limit number of chunks per sample (0 = all chunks)")
    
    args = parser.parse_args()
    
    # Load RAG chunks and query using the same approach as rag_chunk_sparsity.py
    if args.chunks_json:
        is_dataset, samples = load_single_or_dataset(args.chunks_json, args.query)
        if args.sample_limit and is_dataset:
            samples = samples[:args.sample_limit]
            print(f"Limited to first {args.sample_limit} samples from dataset")
        
        if is_dataset:
            # Dataset mode: process multiple samples
            print(f"Processing {len(samples)} samples from dataset")
            
            # Initialize analyzer
            analyzer = RAGChunkCoverageAnalyzer(args.model, args.device)
            
            # Create outputs directory
            os.makedirs(args.out_dir, exist_ok=True)
            
            all_results = []
            
            # Process each sample
            for idx, (chunks, query) in enumerate(samples, 1):
                print(f"\n{'='*60}")
                print(f"Processing Sample {idx}/{len(samples)}")
                print(f"{'='*60}")
                
                # Apply chunk limit if specified
                original_chunk_count = len(chunks)
                if args.chunk_limit > 0 and len(chunks) > args.chunk_limit:
                    chunks = chunks[:args.chunk_limit]
                    print(f"Limited chunks from {original_chunk_count} to {len(chunks)}")
                
                print(f"Chunks: {len(chunks)}")
                print(f"Query: {query}")
                print(f"Attention threshold: {args.attention_threshold}")
                
                # Run analysis for this sample
                print("Running coverage analysis...")
                results = analyzer.analyze_rag_coverage(
                    chunks, query, args.max_new_tokens, args.attention_threshold
                )
                
                # Save individual sample results
                sample_results_path = os.path.join(args.out_dir, f"coverage_analysis_sample_{idx:03d}.json")
                analyzer.save_analysis_results(results, sample_results_path)
                print(f"Saved sample results to: {sample_results_path}")
                
                # Create individual coverage plot
                sample_plot_path = os.path.join(args.out_dir, f"cumulative_coverage_plot_sample_{idx:03d}.png")
                analyzer.create_coverage_plot(results, sample_plot_path)
                print(f"Saved sample plot to: {sample_plot_path}")
                
                # Print summary for this sample
                analyzer.print_coverage_summary(results)
                
                all_results.append(results)
            
            # Save combined results
            combined_results_path = os.path.join(args.out_dir, "coverage_analysis_combined.json")
            combined_data = {
                "num_samples": len(all_results),
                "samples": all_results
            }
            with open(combined_results_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
            print(f"\nSaved combined results to: {combined_results_path}")
            
            # Create combined coverage plot for first 5 samples
            combined_plot_path = os.path.join(args.out_dir, "cumulative_coverage_plot_combined.png")
            analyzer.create_combined_coverage_plot(all_results, combined_plot_path, max_samples=5)
            print(f"Saved combined coverage plot to: {combined_plot_path}")
            
        else:
            # Single sample mode
            chunks, query = samples[0]
            print(f"Loaded single sample with {len(chunks)} chunks")
            
            # Apply chunk limit if specified
            original_chunk_count = len(chunks)
            if args.chunk_limit > 0 and len(chunks) > args.chunk_limit:
                chunks = chunks[:args.chunk_limit]
                print(f"Limited chunks from {original_chunk_count} to {len(chunks)}")
            
            print(f"Loaded {len(chunks)} RAG chunks")
            print(f"Query: {query}")
            print(f"Attention threshold: {args.attention_threshold}")
            
            # Initialize analyzer
            analyzer = RAGChunkCoverageAnalyzer(args.model, args.device)
            
            # Run analysis
            print("Running coverage analysis...")
            results = analyzer.analyze_rag_coverage(
                chunks, query, args.max_new_tokens, args.attention_threshold
            )
            
            # Create outputs
            os.makedirs(args.out_dir, exist_ok=True)
            
            # Save results
            results_path = os.path.join(args.out_dir, "coverage_analysis.json")
            analyzer.save_analysis_results(results, results_path)
            print(f"Saved analysis results to: {results_path}")
            
            # Create coverage plot
            plot_path = os.path.join(args.out_dir, "cumulative_coverage_plot.png")
            analyzer.create_coverage_plot(results, plot_path)
            print(f"Saved coverage plot to: {plot_path}")
            
            # Print summary
            analyzer.print_coverage_summary(results)
    elif args.chunks_dir:
        chunk_files = sorted(Path(args.chunks_dir).glob("*.txt"))
        chunks = [f.read_text().strip() for f in chunk_files]
        query = args.query or "Please provide a comprehensive summary of the given documents."
        
        # Apply chunk limit if specified
        original_chunk_count = len(chunks)
        if args.chunk_limit > 0 and len(chunks) > args.chunk_limit:
            chunks = chunks[:args.chunk_limit]
            print(f"Limited chunks from {original_chunk_count} to {len(chunks)}")
        
        print(f"Loaded {len(chunks)} RAG chunks")
        print(f"Query: {query}")
        print(f"Attention threshold: {args.attention_threshold}")
        
        # Initialize analyzer
        analyzer = RAGChunkCoverageAnalyzer(args.model, args.device)
        
        # Run analysis
        print("Running coverage analysis...")
        results = analyzer.analyze_rag_coverage(
            chunks, query, args.max_new_tokens, args.attention_threshold
        )
        
        # Create outputs
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Save results
        results_path = os.path.join(args.out_dir, "coverage_analysis.json")
        analyzer.save_analysis_results(results, results_path)
        print(f"Saved analysis results to: {results_path}")
        
        # Create coverage plot
        plot_path = os.path.join(args.out_dir, "cumulative_coverage_plot.png")
        analyzer.create_coverage_plot(results, plot_path)
        print(f"Saved coverage plot to: {plot_path}")
        
        # Print summary
        analyzer.print_coverage_summary(results)
    else:
        print("Error: Must provide either --chunks-json or --chunks-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
