#!/usr/bin/env python3
"""
SkyReels-V2 vs Wan Performance Benchmarking Suite

This script provides comprehensive benchmarking to identify performance gaps
between SkyReels-V2 and Wan pipelines, focusing on specific bottlenecks.
"""

import time
import torch
import psutil
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import json
import numpy as np
from pathlib import Path

# Import the pipelines
from diffusers import (
    SkyReelsV2DiffusionForcingPipeline,
    WanPipeline,
    AutoencoderKLWan,
    UniPCMultistepScheduler
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    model_name: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    total_time: float
    memory_peak: float
    memory_allocated: float
    component_times: Dict[str, float]
    fps: float
    throughput: float  # frames per second


class PerformanceProfiler:
    """Advanced profiling context manager"""

    def __init__(self, name: str, enable_cuda_profiling: bool = True):
        self.name = name
        self.enable_cuda_profiling = enable_cuda_profiling
        self.start_time = 0
        self.component_times = {}
        self.memory_stats = {}

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self.start_time = time.time()
        self.start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time

        if torch.cuda.is_available():
            self.memory_peak = torch.cuda.max_memory_allocated()
            self.memory_allocated = torch.cuda.memory_allocated()

    @contextmanager
    def profile_component(self, component_name: str):
        """Profile individual pipeline components"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        self.component_times[component_name] = end_time - start_time


class SkyReelsV2Profiler:
    """Specialized profiler for SkyReels-V2 pipeline components"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.component_times = {}

    def profile_timestep_matrix_generation(self, config: BenchmarkConfig):
        """Profile the generate_timestep_matrix method"""
        num_latent_frames = (config.num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1

        # Create dummy timesteps
        timesteps = torch.linspace(1000, 0, config.num_inference_steps)

        with PerformanceProfiler("timestep_matrix_generation") as profiler:
            step_matrix, _, step_update_mask, valid_interval = self.pipeline.generate_timestep_matrix(
                num_latent_frames=num_latent_frames,
                step_template=timesteps,
                base_num_latent_frames=num_latent_frames,
                ar_step=5,
                num_pre_ready=0,
                causal_block_size=5,
            )

        return profiler.total_time, step_matrix.numel()

    def profile_transformer_calls(self, config: BenchmarkConfig):
        """Profile transformer forward passes"""
        batch_size = config.batch_size
        num_latent_frames = (config.num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1
        latent_height = config.height // self.pipeline.vae_scale_factor_spatial
        latent_width = config.width // self.pipeline.vae_scale_factor_spatial

        # Create dummy inputs
        hidden_states = torch.randn(
            batch_size, 16, num_latent_frames, latent_height, latent_width,
            device=config.device, dtype=config.dtype
        )
        timestep = torch.randint(0, 1000, (batch_size, num_latent_frames), device=config.device)
        encoder_hidden_states = torch.randn(batch_size, 226, 2048, device=config.device, dtype=config.dtype)
        fps_embeds = [24] * batch_size

        # Profile single transformer call
        with PerformanceProfiler("single_transformer_call") as profiler:
            with torch.no_grad():
                noise_pred = self.pipeline.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    enable_diffusion_forcing=True,
                    fps=fps_embeds,
                    return_dict=False,
                )[0]

        return profiler.total_time, profiler.memory_peak


class PipelineBenchmark:
    """Main benchmarking class"""

    def __init__(self):
        self.results = {}

    def load_skyreels_pipeline(self, model_id: str, device: str = "cuda"):
        """Load SkyReels-V2 pipeline"""
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16
        )
        flow_shift = 8.0
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        return pipe.to(device)

    def load_wan_pipeline(self, model_id: str, device: str = "cuda"):
        """Load Wan pipeline"""
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16
        )
        flow_shift = 5.0
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        return pipe.to(device)

    def benchmark_pipeline(self, pipeline, config: BenchmarkConfig, prompt: str) -> BenchmarkResult:
        """Benchmark a single pipeline configuration"""

        with PerformanceProfiler(f"full_pipeline_{config.model_name}") as profiler:

            if "SkyReels" in config.model_name:
                # SkyReels-V2 specific parameters
                with profiler.profile_component("skyreels_generation"):
                    output = pipeline(
                        prompt=prompt,
                        num_inference_steps=config.num_inference_steps,
                        height=config.height,
                        width=config.width,
                        guidance_scale=config.guidance_scale,
                        num_frames=config.num_frames,
                        ar_step=5,
                        causal_block_size=5,
                        addnoise_condition=20,
                    ).frames[0]
            else:
                # Wan pipeline
                with profiler.profile_component("wan_generation"):
                    output = pipeline(
                        prompt=prompt,
                        height=config.height,
                        width=config.width,
                        num_frames=config.num_frames,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                    ).frames[0]

        fps = config.num_frames / profiler.total_time
        throughput = config.num_frames * config.batch_size / profiler.total_time

        return BenchmarkResult(
            total_time=profiler.total_time,
            memory_peak=profiler.memory_peak,
            memory_allocated=profiler.memory_allocated,
            component_times=profiler.component_times,
            fps=fps,
            throughput=throughput
        )

    def run_component_benchmarks(self, config: BenchmarkConfig):
        """Run detailed component-level benchmarks for SkyReels-V2"""

        # Load pipeline for component testing
        pipeline = self.load_skyreels_pipeline(config.model_name, config.device)
        skyreels_profiler = SkyReelsV2Profiler(pipeline)

        results = {}

        # Profile timestep matrix generation
        matrix_time, matrix_size = skyreels_profiler.profile_timestep_matrix_generation(config)
        results["timestep_matrix"] = {
            "time": matrix_time,
            "matrix_size": matrix_size,
            "complexity_estimate": matrix_size * config.num_inference_steps
        }

        # Profile transformer calls
        transformer_time, transformer_memory = skyreels_profiler.profile_transformer_calls(config)
        results["transformer"] = {
            "single_call_time": transformer_time,
            "memory_usage": transformer_memory,
            "estimated_total_calls": config.num_inference_steps * 2  # CFG
        }

        return results

    def run_comparative_benchmark(self, configs: List[BenchmarkConfig], prompt: str):
        """Run comparative benchmarks between SkyReels-V2 and Wan"""

        results = {}

        for config in configs:
            print(f"\\nBenchmarking {config.model_name}...")
            print(f"Resolution: {config.width}x{config.height}, Frames: {config.num_frames}")

            try:
                # Load appropriate pipeline
                if "SkyReels" in config.model_name:
                    pipeline = self.load_skyreels_pipeline(config.model_name, config.device)
                    # Also run component benchmarks
                    component_results = self.run_component_benchmarks(config)
                else:
                    pipeline = self.load_wan_pipeline(config.model_name, config.device)
                    component_results = {}

                # Run full pipeline benchmark
                result = self.benchmark_pipeline(pipeline, config, prompt)

                results[config.model_name] = {
                    "config": config,
                    "result": result,
                    "components": component_results
                }

                # Print results
                print(f"Total time: {result.total_time:.2f}s")
                print(f"FPS: {result.fps:.2f}")
                print(f"Peak memory: {result.memory_peak / 1e9:.2f} GB")

                # Clean up
                del pipeline
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error benchmarking {config.model_name}: {e}")
                results[config.model_name] = {"error": str(e)}

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file"""

        def serialize_result(obj):
            if isinstance(obj, torch.dtype):
                return str(obj)
            elif isinstance(obj, BenchmarkConfig):
                return obj.__dict__
            elif isinstance(obj, BenchmarkResult):
                return obj.__dict__
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_result)

    def generate_performance_report(self, results: Dict) -> str:
        """Generate a detailed performance analysis report"""

        report = ["# SkyReels-V2 vs Wan Performance Analysis\\n"]

        skyreels_results = {k: v for k, v in results.items() if "SkyReels" in k}
        wan_results = {k: v for k, v in results.items() if "Wan" in k}

        if skyreels_results and wan_results:
            report.append("## Performance Comparison\\n")

            for skyreels_key in skyreels_results:
                skyreels_result = skyreels_results[skyreels_key]["result"]

                # Find comparable Wan result
                comparable_wan = None
                for wan_key in wan_results:
                    wan_result = wan_results[wan_key]["result"]
                    if wan_result:  # Basic comparison for now
                        comparable_wan = wan_result
                        break

                if comparable_wan:
                    speedup = comparable_wan.total_time / skyreels_result.total_time
                    memory_ratio = skyreels_result.memory_peak / comparable_wan.memory_peak

                    report.append(f"### {skyreels_key} vs Wan\\n")
                    report.append(f"- **Speed**: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}\\n")
                    report.append(f"- **Memory**: {memory_ratio:.2f}x {'more' if memory_ratio > 1 else 'less'}\\n")
                    report.append(f"- **SkyReels FPS**: {skyreels_result.fps:.2f}\\n")
                    report.append(f"- **Wan FPS**: {comparable_wan.fps:.2f}\\n\\n")

        # Component analysis for SkyReels
        report.append("## SkyReels-V2 Component Analysis\\n")
        for model_name, model_results in skyreels_results.items():
            if "components" in model_results and model_results["components"]:
                components = model_results["components"]
                report.append(f"### {model_name}\\n")

                if "timestep_matrix" in components:
                    tm = components["timestep_matrix"]
                    report.append(f"- **Timestep Matrix Generation**: {tm['time']:.4f}s\\n")
                    report.append(f"- **Matrix Complexity**: {tm['complexity_estimate']} operations\\n")

                if "transformer" in components:
                    tr = components["transformer"]
                    report.append(f"- **Single Transformer Call**: {tr['single_call_time']:.4f}s\\n")
                    report.append(f"- **Estimated Total Transformer Time**: {tr['single_call_time'] * tr['estimated_total_calls']:.2f}s\\n\\n")

        # Optimization recommendations
        report.append("## Optimization Recommendations\\n")
        report.append("Based on the profiling results:\\n\\n")

        if skyreels_results:
            report.append("### High Priority\\n")
            report.append("1. **Timestep Matrix Optimization**: Pre-compute and cache common patterns\\n")
            report.append("2. **Transformer Call Batching**: Combine CFG calls into single forward pass\\n")
            report.append("3. **Memory Management**: Reduce peak memory usage through in-place operations\\n\\n")

            report.append("### Medium Priority\\n")
            report.append("4. **Causal Block Vectorization**: Replace sequential processing with vectorized ops\\n")
            report.append("5. **CUDA Kernel Optimization**: Custom kernels for frequent operations\\n")
            report.append("6. **Compilation**: Use torch.compile for performance-critical methods\\n")

        return "".join(report)


def main():
    """Main benchmarking execution"""

    # Configuration for different test scenarios
    test_configs = [
        # Short video tests
        BenchmarkConfig(
            model_name="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            height=544, width=960, num_frames=49,
            num_inference_steps=20, guidance_scale=6.0
        ),
        # Medium video tests
        BenchmarkConfig(
            model_name="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            height=544, width=960, num_frames=97,
            num_inference_steps=30, guidance_scale=6.0
        ),
        # Wan comparison (adjust model path as needed)
        BenchmarkConfig(
            model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            height=720, width=1280, num_frames=81,
            num_inference_steps=30, guidance_scale=5.0
        ),
    ]

    prompt = ("A cat and a dog baking a cake together in a kitchen. "
              "The cat is carefully measuring flour, while the dog is stirring "
              "the batter with a wooden spoon. The kitchen is cozy, with sunlight "
              "streaming through the window.")

    # Run benchmarks
    benchmark = PipelineBenchmark()
    results = benchmark.run_comparative_benchmark(test_configs, prompt)

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"skyreels_vs_wan_benchmark_{timestamp}.json"
    benchmark.save_results(results, str(results_file))

    # Generate report
    report = benchmark.generate_performance_report(results)
    report_file = output_dir / f"performance_report_{timestamp}.md"

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\\nResults saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    print("\\n" + report)


if __name__ == "__main__":
    main()
