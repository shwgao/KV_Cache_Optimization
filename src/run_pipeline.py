#!/usr/bin/env python3
"""
Main execution script for CacheBlend Pipeline
Demonstrates the complete workflow with example data
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from main_pipeline import CacheBlendPipeline, PipelineConfig

logger = logging.getLogger(__name__)

def create_sample_chunks() -> List[Dict[str, Any]]:
    """Create sample chunks for demonstration"""
    return [
        {
            "id": "chunk_1",
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computers to perform tasks without explicit instructions. It involves training models on data to make predictions or decisions.",
            "title": "Machine Learning Basics",
            "score": 0.95
        },
        {
            "id": "chunk_2", 
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in computer vision, natural language processing, and speech recognition.",
            "title": "Deep Learning Overview",
            "score": 0.88
        },
        {
            "id": "chunk_3",
            "text": "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models to understand, interpret, and generate human language.",
            "title": "Natural Language Processing",
            "score": 0.82
        },
        {
            "id": "chunk_4",
            "text": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world. It involves developing algorithms to process, analyze, and understand digital images and videos.",
            "title": "Computer Vision Fundamentals",
            "score": 0.75
        },
        {
            "id": "chunk_5",
            "text": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward. It is inspired by how humans and animals learn through trial and error.",
            "title": "Reinforcement Learning",
            "score": 0.70
        },
        {
            "id": "chunk_6",
            "text": "Transformers are a type of neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms to process sequences of data and have been the foundation for models like BERT, GPT, and T5.",
            "title": "Transformer Architecture",
            "score": 0.65
        },
        {
            "id": "chunk_7",
            "text": "Attention mechanisms in neural networks allow models to focus on specific parts of the input when making predictions. This has been particularly important in sequence-to-sequence models and has led to significant improvements in machine translation and text generation.",
            "title": "Attention Mechanisms",
            "score": 0.60
        },
        {
            "id": "chunk_8",
            "text": "Large language models (LLMs) are neural networks with billions of parameters trained on vast amounts of text data. They can perform a wide range of natural language tasks including text generation, translation, summarization, and question answering.",
            "title": "Large Language Models",
            "score": 0.55
        }
    ]

def load_chunks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "chunks" in data:
            return data["chunks"]
        else:
            logger.error(f"Invalid format in {file_path}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to load chunks from {file_path}: {e}")
        return []

def save_results(results: Dict[str, Any], output_file: str):
    """Save pipeline results to file"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def run_single_query(
    pipeline: CacheBlendPipeline,
    query: str,
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run pipeline for a single query"""
    
    logger.info(f"Processing query: {query}")
    
    try:
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline(query, chunks)
        
        logger.info("Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {"error": str(e)}

def run_batch_queries(
    pipeline: CacheBlendPipeline,
    queries: List[str],
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Run pipeline for multiple queries"""
    
    results = []
    
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}")
        
        result = run_single_query(pipeline, query, chunks)
        results.append(result)
        
        # Small delay between queries
        import time
        time.sleep(1)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="CacheBlend Pipeline Runner")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name to use"
    )
    
    parser.add_argument(
        "--chunks-file",
        type=str,
        help="Path to JSON file containing chunks"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    
    parser.add_argument(
        "--queries-file",
        type=str,
        help="Path to file containing multiple queries (one per line)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/pipeline_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=["h100", "a100", "default"],
        default="default",
        help="GPU type for optimized configuration"
    )
    
    parser.add_argument(
        "--max-gpu-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to store in GPU"
    )
    
    parser.add_argument(
        "--colbert-model",
        type=str,
        default="colbert-ai/colbert-v2.0",
        help="ColBERT model path for retrieval"
    )
    
    parser.add_argument(
        "--colbert-index",
        type=str,
        help="ColBERT index path (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize configuration
        config_manager = ConfigManager(args.config)
        
        # Apply GPU-specific configuration
        if args.gpu_type == "h100":
            config_manager = config_manager.create_h100_config()
        elif args.gpu_type == "a100":
            config_manager = config_manager.create_a100_config()
        
        # Override model name if specified
        if args.model:
            config_manager.update_config({
                "model": {"model_name": args.model}
            })
        
        # Override max GPU chunks if specified
        if args.max_gpu_chunks:
            config_manager.update_config({
                "cache": {"max_gpu_chunks": args.max_gpu_chunks}
            })
        
        # Override ColBERT settings if specified
        if args.colbert_model or args.colbert_index:
            colbert_updates = {}
            if args.colbert_model:
                colbert_updates["colbert_model_path"] = args.colbert_model
            if args.colbert_index:
                colbert_updates["colbert_index_path"] = args.colbert_index
            
            # Update pipeline config
            config_manager.update_config(colbert_updates)
        
        # Validate configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return 1
        
        # Print configuration
        config_manager.print_config()
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(**config_manager.get_pipeline_config())
        
        # Initialize pipeline
        logger.info("Initializing CacheBlend pipeline...")
        pipeline = CacheBlendPipeline(pipeline_config)
        
        # Load chunks
        if args.chunks_file:
            chunks = load_chunks_from_file(args.chunks_file)
        else:
            logger.info("Using sample chunks")
            chunks = create_sample_chunks()
        
        if not chunks:
            logger.error("No chunks available")
            return 1
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Process queries
        if args.query:
            # Single query
            results = run_single_query(pipeline, args.query, chunks)
            all_results = [results]
            
        elif args.queries_file:
            # Multiple queries from file
            try:
                with open(args.queries_file, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]
                
                logger.info(f"Loaded {len(queries)} queries from {args.queries_file}")
                all_results = run_batch_queries(pipeline, queries, chunks)
                
            except Exception as e:
                logger.error(f"Failed to load queries from {args.queries_file}: {e}")
                return 1
        else:
            # Default query
            default_query = "What is machine learning and how does it relate to artificial intelligence?"
            logger.info(f"Using default query: {default_query}")
            results = run_single_query(pipeline, default_query, chunks)
            all_results = [results]
        
        # Save results
        save_results(all_results, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        for i, result in enumerate(all_results):
            if "error" in result:
                print(f"Query {i+1}: FAILED - {result['error']}")
            else:
                print(f"Query {i+1}: SUCCESS")
                print(f"  Query: {result['query']}")
                print(f"  Final Response: {result['final_response'][:100]}...")
                print(f"  Cache Hit Rate: {result['cache_stats']['hit_rate']:.2%}")
                print(f"  GPU Chunks: {result['cache_stats']['gpu_chunks']}")
                print(f"  CPU Chunks: {result['cache_stats']['cpu_chunks']}")
        
        print(f"\nResults saved to: {args.output}")
        print("="*60)
        
        # Cleanup
        pipeline.cleanup()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())


