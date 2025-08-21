#!/usr/bin/env python3
"""
SkyReels-V2 Real-Time Performance Monitor

This script provides continuous monitoring of SkyReels-V2 performance
during inference to identify bottlenecks in real-time.
"""

import time
import threading
import torch
import psutil
import GPUtil
from collections import deque, defaultdict
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_percent: float
    current_operation: str
    latency_ms: float
    throughput_fps: Optional[float] = None
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0


class RealTimeProfiler:
    """Real-time performance monitoring and profiling"""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 samples
        self.operation_timings: defaultdict = defaultdict(list)
        self.is_monitoring = False
        self.current_operation = "idle"
        self.monitor_thread: Optional[threading.Thread] = None

        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = 0  # Assume single GPU for now
            self.gpus = GPUtil.getGPUs()

    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Real-time monitoring started...")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("Real-time monitoring stopped.")

    def _monitor_loop(self):
        """Main monitoring loop running in background thread"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)

        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        memory_allocated_mb = 0.0
        memory_cached_mb = 0.0

        if self.gpu_available:
            try:
                if self.gpus:
                    gpu = self.gpus[self.gpu_device]
                    gpu_utilization = gpu.load * 100
                    gpu_memory_used = gpu.memoryUsed
                    gpu_memory_total = gpu.memoryTotal

                # PyTorch memory stats
                memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                memory_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024

            except Exception as e:
                print(f"GPU metrics error: {e}")

        return PerformanceMetrics(
            timestamp=timestamp,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_percent=cpu_percent,
            current_operation=self.current_operation,
            latency_ms=0.0,  # Will be set by operation context
            memory_allocated_mb=memory_allocated_mb,
            memory_cached_mb=memory_cached_mb
        )

    def operation_context(self, operation_name: str):
        """Context manager for timing operations"""
        class OperationContext:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = 0.0

            def __enter__(self):
                self.profiler.current_operation = self.name
                self.start_time = time.time()
                if self.profiler.gpu_available:
                    torch.cuda.synchronize()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.profiler.gpu_available:
                    torch.cuda.synchronize()
                end_time = time.time()
                latency_ms = (end_time - self.start_time) * 1000

                # Record timing
                self.profiler.operation_timings[self.name].append(latency_ms)

                # Update current metrics with latency
                if self.profiler.metrics_history:
                    self.profiler.metrics_history[-1].latency_ms = latency_ms

                self.profiler.current_operation = "idle"

        return OperationContext(self, operation_name)

    def get_operation_stats(self, operation_name: str) -> Dict:
        """Get statistics for a specific operation"""
        timings = self.operation_timings[operation_name]
        if not timings:
            return {}

        return {
            'count': len(timings),
            'mean_ms': np.mean(timings),
            'std_ms': np.std(timings),
            'min_ms': np.min(timings),
            'max_ms': np.max(timings),
            'p50_ms': np.percentile(timings, 50),
            'p95_ms': np.percentile(timings, 95),
            'p99_ms': np.percentile(timings, 99)
        }

    def get_current_status(self) -> Dict:
        """Get current system status"""
        if not self.metrics_history:
            return {'status': 'no_data'}

        latest = self.metrics_history[-1]

        # Calculate recent averages (last 10 samples)
        recent_metrics = list(self.metrics_history)[-10:]

        return {
            'current_operation': latest.current_operation,
            'gpu_utilization': latest.gpu_utilization,
            'gpu_memory_percent': (latest.gpu_memory_used / latest.gpu_memory_total * 100) if latest.gpu_memory_total > 0 else 0,
            'cpu_percent': latest.cpu_percent,
            'memory_allocated_mb': latest.memory_allocated_mb,
            'avg_gpu_util': np.mean([m.gpu_utilization for m in recent_metrics]),
            'avg_cpu_util': np.mean([m.cpu_percent for m in recent_metrics]),
            'operations_count': len(self.operation_timings)
        }

    def save_metrics(self, filepath: Path):
        """Save collected metrics to file"""
        data = {
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'gpu_utilization': m.gpu_utilization,
                    'gpu_memory_used': m.gpu_memory_used,
                    'gpu_memory_total': m.gpu_memory_total,
                    'cpu_percent': m.cpu_percent,
                    'current_operation': m.current_operation,
                    'latency_ms': m.latency_ms,
                    'memory_allocated_mb': m.memory_allocated_mb,
                    'memory_cached_mb': m.memory_cached_mb
                }
                for m in self.metrics_history
            ],
            'operation_timings': dict(self.operation_timings)
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_performance_plots(self, output_dir: Path):
        """Generate performance visualization plots"""
        output_dir.mkdir(exist_ok=True)

        if not self.metrics_history:
            print("No metrics data to plot")
            return

        metrics = list(self.metrics_history)
        timestamps = [m.timestamp - metrics[0].timestamp for m in metrics]  # Relative timestamps

        # GPU Utilization Plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        gpu_utils = [m.gpu_utilization for m in metrics]
        plt.plot(timestamps, gpu_utils, 'b-', alpha=0.7)
        plt.title('GPU Utilization %')
        plt.ylabel('Utilization %')
        plt.grid(True, alpha=0.3)

        # GPU Memory Usage
        plt.subplot(2, 2, 2)
        gpu_memory = [m.gpu_memory_used for m in metrics]
        plt.plot(timestamps, gpu_memory, 'r-', alpha=0.7)
        plt.title('GPU Memory Usage (MB)')
        plt.ylabel('Memory (MB)')
        plt.grid(True, alpha=0.3)

        # CPU Usage
        plt.subplot(2, 2, 3)
        cpu_usage = [m.cpu_percent for m in metrics]
        plt.plot(timestamps, cpu_usage, 'g-', alpha=0.7)
        plt.title('CPU Usage %')
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU %')
        plt.grid(True, alpha=0.3)

        # PyTorch Memory
        plt.subplot(2, 2, 4)
        torch_memory = [m.memory_allocated_mb for m in metrics]
        plt.plot(timestamps, torch_memory, 'm-', alpha=0.7, label='Allocated')
        torch_cached = [m.memory_cached_mb for m in metrics]
        plt.plot(timestamps, torch_cached, 'c-', alpha=0.7, label='Cached')
        plt.title('PyTorch Memory Usage')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'system_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Operation Timings Plot
        if self.operation_timings:
            plt.figure(figsize=(14, 6))

            operations = list(self.operation_timings.keys())
            for i, op in enumerate(operations):
                timings = self.operation_timings[op]
                if timings:
                    plt.subplot(1, len(operations), i + 1)
                    plt.hist(timings, bins=20, alpha=0.7, edgecolor='black')
                    plt.title(f'{op}\\nMean: {np.mean(timings):.2f}ms')
                    plt.xlabel('Latency (ms)')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'operation_timings.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Performance plots saved to {output_dir}")


class SkyReelsV2Monitor:
    """Specialized monitor for SkyReels-V2 pipeline"""

    def __init__(self, pipeline, profiler: RealTimeProfiler):
        self.pipeline = pipeline
        self.profiler = profiler
        self.step_timings = []
        self.component_timings = defaultdict(list)

        # Wrap key methods with monitoring
        self._wrap_pipeline_methods()

    def _wrap_pipeline_methods(self):
        """Wrap pipeline methods with performance monitoring"""

        # Store original methods
        self.original_generate_timestep_matrix = self.pipeline.generate_timestep_matrix
        self.original_transformer_call = self.pipeline.transformer

        # Wrap generate_timestep_matrix
        def monitored_generate_timestep_matrix(*args, **kwargs):
            with self.profiler.operation_context("generate_timestep_matrix"):
                return self.original_generate_timestep_matrix(*args, **kwargs)

        self.pipeline.generate_timestep_matrix = monitored_generate_timestep_matrix

        # Wrap transformer forward pass
        original_forward = self.pipeline.transformer.forward
        def monitored_transformer_forward(*args, **kwargs):
            with self.profiler.operation_context("transformer_forward"):
                return original_forward(*args, **kwargs)

        self.pipeline.transformer.forward = monitored_transformer_forward

    def monitor_inference_step(self, step_idx: int, total_steps: int):
        """Monitor individual inference steps"""

        with self.profiler.operation_context(f"inference_step_{step_idx}"):
            # This would be called during each denoising step
            status = self.profiler.get_current_status()

            # Log significant resource usage
            if status['gpu_utilization'] > 90:
                print(f"⚠️  High GPU utilization at step {step_idx}: {status['gpu_utilization']:.1f}%")

            if status['gpu_memory_percent'] > 85:
                print(f"⚠️  High GPU memory usage at step {step_idx}: {status['gpu_memory_percent']:.1f}%")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""

        summary = {
            'system_status': self.profiler.get_current_status(),
            'operation_stats': {}
        }

        # Get stats for each operation
        for op_name in self.profiler.operation_timings:
            summary['operation_stats'][op_name] = self.profiler.get_operation_stats(op_name)

        # Calculate bottleneck analysis
        total_time = sum(
            stats['mean_ms'] * stats['count']
            for stats in summary['operation_stats'].values()
        )

        if total_time > 0:
            summary['bottleneck_analysis'] = {}
            for op_name, stats in summary['operation_stats'].items():
                op_total = stats['mean_ms'] * stats['count']
                percentage = (op_total / total_time) * 100
                summary['bottleneck_analysis'][op_name] = {
                    'total_time_ms': op_total,
                    'percentage': percentage,
                    'calls_per_second': stats['count'] / (total_time / 1000) if total_time > 0 else 0
                }

        return summary

    def print_live_stats(self):
        """Print real-time statistics"""
        summary = self.get_performance_summary()
        system = summary['system_status']

        print("\\n" + "=" * 60)
        print("🚀 SkyReels-V2 Live Performance Monitor")
        print("=" * 60)
        print(f"Current Operation: {system['current_operation']}")
        print(f"GPU Utilization: {system['gpu_utilization']:.1f}% (avg: {system['avg_gpu_util']:.1f}%)")
        print(f"GPU Memory: {system['gpu_memory_percent']:.1f}%")
        print(f"CPU Usage: {system['cpu_percent']:.1f}% (avg: {system['avg_cpu_util']:.1f}%)")
        print(f"PyTorch Memory: {system['memory_allocated_mb']:.1f} MB")

        if summary.get('bottleneck_analysis'):
            print("\\n📊 Top Time Consumers:")
            bottlenecks = sorted(
                summary['bottleneck_analysis'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )[:5]

            for op_name, data in bottlenecks:
                print(f"  {op_name}: {data['percentage']:.1f}% ({data['total_time_ms']:.1f}ms total)")

        print("=" * 60)


def create_monitoring_pipeline(model_id: str) -> tuple:
    """Create monitored SkyReels-V2 pipeline"""

    # Import here to avoid circular imports during testing
    from diffusers import SkyReelsV2DiffusionForcingPipeline, AutoencoderKLWan, UniPCMultistepScheduler

    # Load pipeline
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16
    )
    flow_shift = 8.0
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config, flow_shift=flow_shift
    )
    pipeline = pipeline.to("cuda")

    # Create profiler and monitor
    profiler = RealTimeProfiler(sampling_interval=0.05)  # 20 FPS monitoring
    monitor = SkyReelsV2Monitor(pipeline, profiler)

    return pipeline, profiler, monitor


def run_monitored_inference(model_id: str, prompt: str, output_dir: Path):
    """Run inference with full monitoring"""

    output_dir.mkdir(exist_ok=True)

    # Create monitored pipeline
    pipeline, profiler, monitor = create_monitoring_pipeline(model_id)

    # Start monitoring
    profiler.start_monitoring()

    try:
        print(f"🎬 Starting monitored inference...")
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt[:80]}...")

        # Run inference with monitoring
        with profiler.operation_context("full_inference"):
            output = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                height=544,
                width=960,
                guidance_scale=6.0,
                num_frames=49,
                ar_step=5,
                causal_block_size=5,
                addnoise_condition=20,
            ).frames[0]

        # Print final statistics
        monitor.print_live_stats()

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        profiler.save_metrics(output_dir / f"performance_metrics_{timestamp}.json")
        profiler.generate_performance_plots(output_dir / f"plots_{timestamp}")

        # Save performance summary
        summary = monitor.get_performance_summary()
        with open(output_dir / f"performance_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\\n✅ Monitoring results saved to {output_dir}")

        return output, summary

    finally:
        profiler.stop_monitoring()


if __name__ == "__main__":
    # Example usage
    model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
    prompt = ("A cat and a dog baking a cake together in a kitchen. "
              "The cat is carefully measuring flour, while the dog is stirring "
              "the batter with a wooden spoon. The kitchen is cozy, with sunlight "
              "streaming through the window.")

    output_dir = Path("monitoring_results")

    try:
        video, summary = run_monitored_inference(model_id, prompt, output_dir)
        print("\\n🎉 Monitored inference completed successfully!")

    except Exception as e:
        print(f"❌ Error during monitored inference: {e}")
        import traceback
        traceback.print_exc()
