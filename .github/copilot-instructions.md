# Diffusers AI Coding Agent Instructions

## Overview
Diffusers is a modular PyTorch library for state-of-the-art diffusion models. The codebase prioritizes **usability over performance**, **simple over easy**, and **tweakable/contributor-friendly over abstraction**.

## Core Architecture

### Three Main Components
1. **Pipelines** (`src/diffusers/pipelines/`): End-to-end inference workflows
2. **Models** (`src/diffusers/models/`): Configurable model architectures (UNets, Transformers, VAEs)
3. **Schedulers** (`src/diffusers/schedulers/`): Noise scheduling for inference/training

### Key Design Principles (from PHILOSOPHY.md)

#### Single-File Policy
- **Pipelines and Schedulers**: Each follows strict single-file policy - all logic in one self-contained file
- **Models**: Partial adherence - use shared building blocks (`attention.py`, `resnet.py`, `embeddings.py`)
- Prefer copy-paste over premature abstraction (following Transformers library pattern)
- Use `# Copied from` mechanism for shared functionality between similar pipelines

#### Mixin Pattern
Models inherit from multiple mixins providing specific capabilities:
- `ModelMixin`: Base model functionality with `from_pretrained()` loading
- `ConfigMixin`: Configuration management
- `PeftAdapterMixin`: LoRA and other parameter-efficient fine-tuning support
- `FromOriginalModelMixin`: Load weights from original checkpoint formats
- `CacheMixin`: Memory-efficient caching for transformers
- `AttentionMixin`: Attention processor management

Example from `transformer_magi1.py`:
```python
class Magi1Transformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
```

## Critical Workflows

### Model Checkpoint Conversion
When adding new model support, create conversion scripts in `scripts/convert_*_to_diffusers.py`:

1. **Download from Hub**: Use `hf_hub_download()` or `snapshot_download()` to fetch original weights
2. **State Dict Mapping**: Create key-mapping functions that translate original checkpoint keys to diffusers format
   - Example pattern: `convert_transformer_state_dict()` functions map layer-by-layer
   - Handle QKV splits: `qkv.weight` → separate `to_q.weight`, `to_k.weight`, `to_v.weight`
   - Use `init_empty_weights` context when available to avoid OOM during large model loading
3. **Verification**: Always verify conversion with shape checks and missing key reports
4. **Save Pipeline**: Use `pipe.save_pretrained()` with `safe_serialization=True` and appropriate `max_shard_size`

Example from `convert_magi1_to_diffusers.py`:
```python
from accelerate import init_empty_weights
# Load checkpoint shards from Hub
checkpoint_files = []
for shard in range(num_shards):
    checkpoint_files.append(hf_hub_download(repo_id, filename))

# Convert state dict with strict validation
converted_state_dict, report = convert_transformer_state_dict(checkpoint, transformer)
transformer.load_state_dict(converted_state_dict, strict=True)
```

### Testing Requirements

Tests follow unittest pattern with specific base classes:
- LoRA tests inherit from `PeftLoraLoaderMixinTests` (see `tests/lora/test_lora_layers_*.py`)
- Define `pipeline_class`, `scheduler_cls`, `transformer_cls`, `vae_cls` as class attributes
- Implement `get_dummy_inputs()` for minimal reproducible inputs
- Use `@require_peft_backend` decorator for LoRA tests

Run tests: `python -m pytest -n auto --dist=loadfile -s -v ./tests/`

### Code Quality Checks

Before submitting PRs, run:
```bash
make fixup  # Fast check on modified files only
make quality  # Full quality checks (ruff, doc-builder)
make style  # Auto-format and fix all files
```

Key quality scripts:
- `utils/check_copies.py`: Validates `# Copied from` annotations
- `utils/check_dummies.py`: Ensures dummy objects for optional dependencies
- `utils/check_repo.py`: Repository-wide consistency checks
- `utils/check_inits.py`: Validates `__init__.py` imports

### Environment Setup

Install in editable mode for development:
```bash
pip install -e ".[torch,test]"  # Core + PyTorch + test dependencies
# Set PYTHONPATH for scripts
export PYTHONPATH=src
```

## Project-Specific Conventions

### Pipeline Structure
- All pipelines inherit from `DiffusionPipeline`
- Components stored in `model_index.json` at Hub repo root
- Single entry point via `__call__()` method
- Device management follows PyTorch: use `.to(device)` explicitly
- Default: CPU, float32 (usability over performance)

### Naming Conventions
- Pipelines: Named after task, e.g., `StableDiffusionPipeline`, `Magi1Pipeline`
- Models: Named after architecture type, e.g., `UNet2DConditionModel`, `Magi1Transformer3DModel`
- Conversion scripts: `convert_<source>_to_diffusers.py`
- Example: `MAGI-1-T2V-4.5B-distill` → `Magi1Pipeline` with `Magi1Transformer3DModel`

### VAE and Model Integration
- VAEs are separate components, not embedded in pipelines
- Text encoders from Transformers library (e.g., `UMT5EncoderModel`, `T5EncoderModel`)
- Tokenizers from Transformers library
- Schedulers instantiated with config (e.g., `FlowMatchEulerDiscreteScheduler(shift=3.0)`)

### State Dict Key Mapping Patterns
When converting checkpoints, common patterns:
- `encoder.blocks.{i}.attn.qkv.*` → separate `to_q`, `to_k`, `to_v` projections
- `mlp.fc1/fc2` → `ffn.net.0.proj` / `ffn.net.2`
- `norm2` → `norm` (post-attention norm)
- Ada-LayerNorm: `ada_modulate_layer.proj.0.*` → `ada_modulate_layer.1.*`

### Hub Integration
- Use `push_to_hub=True` in `save_pretrained()` for direct Hub uploads
- Specify `repo_id` for organization/user namespacing
- Use `variant="fp16"` or `variant="bf16"` for reduced precision checkpoints
- `max_shard_size="5GB"` recommended for large models

### Documentation
- Models/Pipelines must have comprehensive docstrings
- Use `src/diffusers/pipelines/<name>/README.md` for pipeline-specific docs
- Example code in docstrings should be minimal and runnable
- Link to papers in class docstrings

## Common Pitfalls

1. **Don't modify existing abstractions unnecessarily** - Follow single-file policy
2. **Always use `strict=True` for `load_state_dict()`** unless explicitly required otherwise
3. **Test with `from_pretrained()` loading** - Verify config serialization roundtrip
4. **Check dtype consistency** - Default float32, explicit conversion for fp16/bf16
5. **Verify shape mismatches** during conversion - Compare expected vs. converted state dict keys
6. **Use `init_empty_weights` for large models** to avoid OOM during meta-device loading
7. **Don't forget `safe_serialization=True`** when saving models

## Examples Directory
- Training examples in `examples/`: self-contained, beginner-friendly, single-purpose
- Each has own `requirements.txt`
- Not feature-complete UIs - focus on educational value
- Active maintenance by core team for official examples

## Quick Reference

**Add new pipeline**: Create in `src/diffusers/pipelines/<name>/pipeline_<name>.py` following single-file policy
**Convert model**: Script in `scripts/convert_<name>_to_diffusers.py` with state dict mapping
**Test LoRA support**: Inherit from `PeftLoraLoaderMixinTests` in `tests/lora/`
**Check quality**: `make fixup` before commits
**Run specific tests**: `python -m pytest tests/pipelines/test_<name>.py -v`

For detailed philosophy, see `PHILOSOPHY.md`. For contribution workflow, see `CONTRIBUTING.md`.
