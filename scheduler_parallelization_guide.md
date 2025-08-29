# Scheduler Parallelization Solutions for SkyReels V2

## Current Problem
The sequential loop processes scheduler steps one frame at a time:
```python
for idx in range(valid_interval_start, valid_interval_end):
    if update_mask_i[idx].item():
        latents[:, :, idx, :, :] = sample_schedulers[idx].step(
            noise_pred[:, :, idx - valid_interval_start, :, :],
            t[idx],
            latents[:, :, idx, :, :],
            return_dict=False,
        )[0]
```

## Solution 1: Threading Approach (Already Implemented)

The current implementation uses `ThreadPoolExecutor` to parallelize scheduler steps. This works well for:
- I/O bound operations
- Python functions with minimal computational overhead
- Small to medium number of frames

**Pros:**
- Simple to implement
- Good for I/O bound operations
- Shared memory access

**Cons:**
- Limited by Python's GIL for CPU-bound tasks
- Memory overhead for threads

## Solution 2: Process-Based Parallelization

For heavy computational workloads, use `ProcessPoolExecutor`:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def scheduler_step_worker(args):
    """Worker function for process-based parallelization"""
    scheduler_state, noise_pred_slice, timestep, latent_slice = args

    # Recreate scheduler from state
    scheduler = deepcopy(sample_scheduler_template)
    scheduler.load_state_dict(scheduler_state)

    result = scheduler.step(
        noise_pred_slice,
        timestep,
        latent_slice,
        return_dict=False,
    )[0]

    return result

# Usage:
with ProcessPoolExecutor(max_workers=mp.cpu_count()//2) as executor:
    tasks = []
    for idx in indices_to_process:
        args = (
            sample_schedulers[idx].state_dict(),
            noise_pred[:, :, idx - valid_interval_start, :, :],
            t[idx],
            latents[:, :, idx, :, :]
        )
        tasks.append(executor.submit(scheduler_step_worker, args))

    for idx, future in zip(indices_to_process, tasks):
        latents[:, :, idx, :, :] = future.result()
```

## Solution 3: Vectorized Batch Processing

Create a batched scheduler step function that processes multiple frames at once:

```python
def batch_scheduler_step(schedulers, noise_preds, timesteps, latents_batch):
    """
    Process multiple scheduler steps in a vectorized manner
    """
    results = []

    # Group by timestep for potential optimization
    timestep_groups = {}
    for i, (scheduler, noise_pred, timestep, latent) in enumerate(
        zip(schedulers, noise_preds, timesteps, latents_batch)
    ):
        if timestep.item() not in timestep_groups:
            timestep_groups[timestep.item()] = []
        timestep_groups[timestep.item()].append((i, scheduler, noise_pred, latent))

    # Process each timestep group
    result_dict = {}
    for timestep, group in timestep_groups.items():
        for i, scheduler, noise_pred, latent in group:
            result = scheduler.step(noise_pred, timestep, latent, return_dict=False)[0]
            result_dict[i] = result

    # Return results in original order
    return [result_dict[i] for i in range(len(schedulers))]
```

## Solution 4: CUDA Streams (GPU Acceleration)

If using CUDA, leverage CUDA streams for parallel GPU execution:

```python
import torch.cuda as cuda

def parallel_scheduler_step_cuda(indices_to_process, streams=None):
    if streams is None:
        streams = [cuda.Stream() for _ in range(min(len(indices_to_process), 4))]

    results = {}

    for i, idx in enumerate(indices_to_process):
        stream = streams[i % len(streams)]

        with cuda.stream(stream):
            # Move tensors to stream
            noise_pred_slice = noise_pred[:, :, idx - valid_interval_start, :, :].cuda(non_blocking=True)
            latent_slice = latents[:, :, idx, :, :].cuda(non_blocking=True)

            # Process scheduler step
            result = sample_schedulers[idx].step(
                noise_pred_slice,
                t[idx],
                latent_slice,
                return_dict=False,
            )[0]

            results[idx] = result

    # Synchronize all streams
    for stream in streams:
        stream.synchronize()

    return results
```

## Solution 5: JIT Compilation with TorchScript

Compile the scheduler step loop for better performance:

```python
import torch.jit as jit

@jit.script
def jit_scheduler_step_loop(
    noise_pred: torch.Tensor,
    latents: torch.Tensor,
    update_mask: torch.Tensor,
    timesteps: torch.Tensor,
    valid_start: int,
    valid_end: int
) -> torch.Tensor:
    """JIT compiled version of the scheduler step loop"""
    for idx in range(valid_start, valid_end):
        if update_mask[idx].item():
            # Note: This would require a JIT-compatible scheduler implementation
            pass  # Scheduler step logic here
    return latents
```

## Solution 6: Memory-Optimized Approach

For very large videos, use memory mapping and chunked processing:

```python
def chunked_scheduler_processing(
    chunk_size: int = 8,  # Process 8 frames at a time
    overlap: int = 1      # Overlap between chunks
):
    """Process schedulers in chunks to optimize memory usage"""

    num_frames = valid_interval_end - valid_interval_start

    for chunk_start in range(0, num_frames, chunk_size - overlap):
        chunk_end = min(chunk_start + chunk_size, num_frames)

        # Process chunk
        chunk_indices = range(
            valid_interval_start + chunk_start,
            valid_interval_start + chunk_end
        )

        # Parallel processing within chunk
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for idx in chunk_indices:
                if update_mask_i[idx].item():
                    future = executor.submit(
                        sample_schedulers[idx].step,
                        noise_pred[:, :, idx - valid_interval_start, :, :],
                        t[idx],
                        latents[:, :, idx, :, :],
                        False
                    )
                    futures[idx] = future

            # Collect results
            for idx, future in futures.items():
                latents[:, :, idx, :, :] = future.result()[0]
```

## Recommended Implementation Strategy

1. **Start with Threading** (already implemented): Good for most cases
2. **Profile your workload**: Measure CPU vs GPU utilization
3. **Consider Process-based**: If CPU-bound and have multiple cores
4. **GPU Optimization**: If using CUDA, implement CUDA streams
5. **Memory-aware**: For very long videos, use chunked processing

## Performance Considerations

- **Thread count**: Start with `min(num_frames, cpu_count())`
- **Memory usage**: Monitor memory consumption with parallel execution
- **GPU memory**: Be careful about GPU memory when parallelizing
- **Scheduler state**: Ensure scheduler independence between parallel executions

## Benchmarking

To measure improvement, add timing:

```python
import time

start_time = time.time()
# Your parallel implementation here
end_time = time.time()

print(f"Sequential time: {sequential_time:.4f}s")
print(f"Parallel time: {end_time - start_time:.4f}s")
print(f"Speedup: {sequential_time / (end_time - start_time):.2f}x")
```
