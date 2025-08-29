#!/usr/bin/env python3
"""
Configuration Management for CacheBlend Pipeline
Centralized configuration handling for all pipeline components
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "meta-llama/Llama-3.1-8B"
    num_layers: int = 32
    num_attention_heads: int = 32
    hidden_size: int = 4096
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    dtype: str = "float16"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    max_gpu_chunks: int = 5
    max_cpu_chunks: int = 50
    gpu_memory_limit_gb: float = 40.0
    cpu_memory_limit_gb: float = 100.0
    safety_margin_gb: float = 2.0
    max_memory_utilization: float = 0.9
    chunk_size: int = 256
    cache_dtype: str = "float16"

@dataclass
class SchedulerConfig:
    """Scheduler configuration settings"""
    max_concurrent_swaps: int = 2
    swap_threshold: float = 0.5
    prediction_weight: float = 0.7
    usage_weight: float = 0.3
    update_interval: float = 0.1

@dataclass
class SpeculativeConfig:
    """Speculative decoding configuration"""
    prediction_window: int = 10
    confidence_threshold: float = 0.7
    max_predictions: int = 5
    learning_rate: float = 0.01

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    log_interval: float = 1.0
    max_history: int = 1000
    output_dir: str = "results"
    save_metrics: bool = True
    print_summary: bool = True

@dataclass
class CacheBlendConfig:
    """Main configuration for CacheBlend pipeline"""
    model: ModelConfig
    cache: CacheConfig
    scheduler: SchedulerConfig
    speculative: SpeculativeConfig
    performance: PerformanceConfig
    device: str = "cuda"
    log_level: str = "INFO"
    seed: int = 42

class ConfigManager:
    """Manages configuration for the CacheBlend pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Configuration manager initialized")
    
    def _load_config(self) -> CacheBlendConfig:
        """Load configuration from file or use defaults"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                return self._dict_to_config(config_dict)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # Use default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> CacheBlendConfig:
        """Get default configuration"""
        return CacheBlendConfig(
            model=ModelConfig(),
            cache=CacheConfig(),
            scheduler=SchedulerConfig(),
            speculative=SpeculativeConfig(),
            performance=PerformanceConfig()
        )
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CacheBlendConfig:
        """Convert dictionary to configuration object"""
        return CacheBlendConfig(
            model=ModelConfig(**config_dict.get("model", {})),
            cache=CacheConfig(**config_dict.get("cache", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
            speculative=SpeculativeConfig(**config_dict.get("speculative", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            device=config_dict.get("device", "cuda"),
            log_level=config_dict.get("log_level", "INFO"),
            seed=config_dict.get("seed", 42)
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('cacheblend.log')
            ]
        )
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get configuration for the main pipeline"""
        return {
            "model_name": self.config.model.model_name,
            "max_gpu_chunks": self.config.cache.max_gpu_chunks,
            "max_cpu_chunks": self.config.cache.max_cpu_chunks,
            "gpu_memory_limit_gb": self.config.cache.gpu_memory_limit_gb,
            "cpu_memory_limit_gb": self.config.cache.cpu_memory_limit_gb,
            "safety_margin_gb": self.config.cache.safety_margin_gb,
            "max_memory_utilization": self.config.cache.max_memory_utilization,
            "device": self.config.device,
            "output_dir": self.config.performance.output_dir
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return asdict(self.config.model)
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return asdict(self.config.cache)
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration"""
        return asdict(self.config.scheduler)
    
    def get_speculative_config(self) -> Dict[str, Any]:
        """Get speculative decoding configuration"""
        return asdict(self.config.speculative)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        return asdict(self.config.performance)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            # Update model config
            if "model" in updates:
                for key, value in updates["model"].items():
                    if hasattr(self.config.model, key):
                        setattr(self.config.model, key, value)
            
            # Update cache config
            if "cache" in updates:
                for key, value in updates["cache"].items():
                    if hasattr(self.config.cache, key):
                        setattr(self.config.cache, key, value)
            
            # Update scheduler config
            if "scheduler" in updates:
                for key, value in updates["scheduler"].items():
                    if hasattr(self.config.scheduler, key):
                        setattr(self.config.scheduler, key, value)
            
            # Update speculative config
            if "speculative" in updates:
                for key, value in updates["speculative"].items():
                    if hasattr(self.config.speculative, key):
                        setattr(self.config.speculative, key, value)
            
            # Update performance config
            if "performance" in updates:
                for key, value in updates["performance"].items():
                    if hasattr(self.config.performance, key):
                        setattr(self.config.performance, key, value)
            
            # Update main config
            for key, value in updates.items():
                if key not in ["model", "cache", "scheduler", "speculative", "performance"]:
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate model config
        if self.config.model.num_layers <= 0:
            errors.append("num_layers must be positive")
        
        if self.config.model.num_attention_heads <= 0:
            errors.append("num_attention_heads must be positive")
        
        if self.config.model.hidden_size <= 0:
            errors.append("hidden_size must be positive")
        
        # Validate cache config
        if self.config.cache.max_gpu_chunks <= 0:
            errors.append("max_gpu_chunks must be positive")
        
        if self.config.cache.max_cpu_chunks <= 0:
            errors.append("max_cpu_chunks must be positive")
        
        if self.config.cache.gpu_memory_limit_gb <= 0:
            errors.append("gpu_memory_limit_gb must be positive")
        
        if self.config.cache.cpu_memory_limit_gb <= 0:
            errors.append("cpu_memory_limit_gb must be positive")
        
        # Validate scheduler config
        if self.config.scheduler.max_concurrent_swaps <= 0:
            errors.append("max_concurrent_swaps must be positive")
        
        if not 0 <= self.config.scheduler.swap_threshold <= 1:
            errors.append("swap_threshold must be between 0 and 1")
        
        if not 0 <= self.config.scheduler.prediction_weight <= 1:
            errors.append("prediction_weight must be between 0 and 1")
        
        if not 0 <= self.config.scheduler.usage_weight <= 1:
            errors.append("usage_weight must be between 0 and 1")
        
        # Validate speculative config
        if self.config.speculative.prediction_window <= 0:
            errors.append("prediction_window must be positive")
        
        if not 0 <= self.config.speculative.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if self.config.speculative.max_predictions <= 0:
            errors.append("max_predictions must be positive")
        
        # Check for errors
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CACHEBLEND CONFIGURATION")
        print("="*60)
        
        config_dict = asdict(self.config)
        for section, settings in config_dict.items():
            print(f"\n{section.upper()}:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {settings}")
        
        print("="*60)
    
    def create_h100_config(self) -> 'ConfigManager':
        """Create configuration optimized for H100 GPU"""
        h100_config = CacheBlendConfig(
            model=ModelConfig(
                model_name="meta-llama/Llama-2-7b-hf",
                num_layers=32,
                num_attention_heads=32,
                hidden_size=4096,
                vocab_size=32000,
                max_position_embeddings=2048,
                dtype="float16"
            ),
            cache=CacheConfig(
                max_gpu_chunks=5,  # Optimized for H100
                max_cpu_chunks=50,
                gpu_memory_limit_gb=80.0,  # H100 has 80GB
                cpu_memory_limit_gb=100.0,
                safety_margin_gb=4.0,
                max_memory_utilization=0.95,
                chunk_size=256,
                cache_dtype="float16"
            ),
            scheduler=SchedulerConfig(
                max_concurrent_swaps=3,
                swap_threshold=0.6,
                prediction_weight=0.8,
                usage_weight=0.2,
                update_interval=0.05
            ),
            speculative=SpeculativeConfig(
                prediction_window=15,
                confidence_threshold=0.8,
                max_predictions=8,
                learning_rate=0.01
            ),
            performance=PerformanceConfig(
                log_interval=0.5,
                max_history=2000,
                output_dir="results_h100",
                save_metrics=True,
                print_summary=True
            ),
            device="cuda",
            log_level="INFO",
            seed=42
        )
        
        new_manager = ConfigManager()
        new_manager.config = h100_config
        return new_manager
    
    def create_a100_config(self) -> 'ConfigManager':
        """Create configuration optimized for A100 GPU"""
        a100_config = CacheBlendConfig(
            model=ModelConfig(
                model_name="meta-llama/Llama-2-7b-hf",
                num_layers=32,
                num_attention_heads=32,
                hidden_size=4096,
                vocab_size=32000,
                max_position_embeddings=2048,
                dtype="float16"
            ),
            cache=CacheConfig(
                max_gpu_chunks=3,  # Optimized for A100
                max_cpu_chunks=30,
                gpu_memory_limit_gb=40.0,  # A100 has 40GB
                cpu_memory_limit_gb=80.0,
                safety_margin_gb=2.0,
                max_memory_utilization=0.9,
                chunk_size=256,
                cache_dtype="float16"
            ),
            scheduler=SchedulerConfig(
                max_concurrent_swaps=2,
                swap_threshold=0.5,
                prediction_weight=0.7,
                usage_weight=0.3,
                update_interval=0.1
            ),
            speculative=SpeculativeConfig(
                prediction_window=10,
                confidence_threshold=0.7,
                max_predictions=5,
                learning_rate=0.01
            ),
            performance=PerformanceConfig(
                log_interval=1.0,
                max_history=1000,
                output_dir="results_a100",
                save_metrics=True,
                print_summary=True
            ),
            device="cuda",
            log_level="INFO",
            seed=42
        )
        
        new_manager = ConfigManager()
        new_manager.config = a100_config
        return new_manager


