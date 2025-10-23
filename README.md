# FLUX.1-schnell Image Generation

A collection of Python scripts for generating images using the FLUX.1-schnell text-to-image diffusion model with comprehensive performance monitoring, batch processing, and cross-platform support.

## Overview

This repository provides multiple versions of image generation scripts that progressively add features for benchmarking, profiling, and batch processing. From simple single-image generation to comprehensive system profiling with JSON export, these scripts are designed for testing FLUX.1-schnell performance across different hardware configurations.

## Features

- **Multi-Platform Support**: CUDA GPUs, Apple Silicon (MPS), and CPU
- **Batch Processing**: Generate multiple images with configurable sub-batching
- **Performance Monitoring**: Detailed timing and memory usage tracking
- **System Profiling**: Comprehensive hardware and software environment capture
- **JSON Export**: Structured performance data for analysis and comparison
- **Device Auto-Detection**: Automatically selects the best available accelerator
- **Memory Management**: Smart VRAM offloading and RAM tracking

## Installation

### macOS (Apple Silicon)

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.7.0 torchvision torchaudio==2.7.0 --index-url https://pypi.apple.com/simple
uv pip install transformers protobuf accelerate diffusers sentencepiece psutil
```

### Ubuntu/Linux (CUDA)

```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install accelerate diffusers transformers protobuf sentencepiece psutil
```

### Windows

```bash
uv venv --python=3.11
.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install accelerate diffusers transformers protobuf sentencepiece psutil
```

Or install from `requirements_windows.txt`:
```bash
uv pip install -r requirements_windows.txt
```

## Quick Start

### Basic Image Generation

```bash
python run_flux.py --prompt "A cat holding a sign that says hello world" --output cat.png
```

### Batch Generation with Performance Tracking

```bash
python run_flux8.py \
  --prompt "Pink elephant in a modern conference room" \
  --guidance_scale 1.5 \
  --num_inference_steps 6 \
  --max_sequence_length 256 \
  --seed 777 \
  --width 1280 \
  --height 720 \
  --batch-size 16 \
  --sub-batch-size 1 \
  --json performance_report.json \
  --output pink_elephant.png
```

## Script Versions

### `run_flux.py` - Basic Generator
Minimal implementation for simple image generation.

**Use case**: Quick single-image generation without overhead

### `run_flux4.py` - Performance Monitor
Adds timing measurements and memory usage tracking.

**Use case**: Basic performance profiling

**Features**:
- Device auto-detection (CUDA/MPS/CPU)
- Timing breakdown (loading, generation, saving)
- Memory usage reporting

### `run_flux5.py` - Batch Processor
Adds support for generating multiple images with sub-batching.

**Use case**: Generating multiple variations or testing reproducibility

**Features**:
- Configurable batch sizes
- Sub-batch processing to manage memory
- Per-image timing
- Unique seeds for each image
- Optional delays between sub-batches

### `run_flux8.py` - Production Profiler (Recommended)
Comprehensive system profiling with JSON export for benchmarking.

**Use case**: Hardware benchmarking, performance analysis, system comparison

**Features**:
- Complete system information capture
- CUDA/driver version detection
- RAM tracking across all phases
- Package version collection
- JSON report generation
- Detailed performance metrics

## Command-Line Options

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prompt` | string | *required* | Text description of the image to generate |
| `--output` | string | `flux-schnell.png` | Base output filename |
| `--width` | int | 1024 | Image width in pixels |
| `--height` | int | 1024 | Image height in pixels |
| `--seed` | int | 0 | Random seed for reproducibility |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--guidance_scale` | float | 0.0 | Classifier-free guidance scale (0.0-2.0) |
| `--num_inference_steps` | int | 4 | Number of denoising steps |
| `--max_sequence_length` | int | 256 | Maximum tokens for text encoding |

### Batch Processing (run_flux5.py, run_flux8.py)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch-size` | int | 1 | Total number of images to generate |
| `--sub-batch-size` | int | 1 | Images per sub-batch iteration |
| `--wait-between-sub-batches` | float | 0.0 | Seconds to wait between sub-batches |

### Performance Tracking (run_flux8.py)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--json` | string | None | Path to save JSON performance report |

### torch.compile Optimization (run_flux8.py)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--compile` | flag | False | Enable torch.compile for transformer and VAE |
| `--compile-mode` | string | `reduce-overhead` | Compilation mode: `default`, `reduce-overhead`, `max-autotune` |
| `--compile-dynamic` | flag | False | Enable dynamic shapes to avoid recompilation on dimension changes |
| `--compile-regional` | flag | False | Use regional compilation for faster compile time |

## Performance Optimization with torch.compile

The `run_flux8.py` script supports `torch.compile` for significant performance improvements. Based on [PyTorch's official guidance for diffusion models](https://pytorch.org/blog/torch-compile-diffusers/), we've implemented multiple compilation strategies.

### Quick Start

Enable basic compilation:

```bash
python run_flux8.py \
  --prompt "A serene mountain landscape" \
  --compile \
  --width 1280 --height 720 \
  --json results.json
```

### macOS Users: Recommended Configuration

**For the best experience on Apple Silicon (M1/M2/M3/M4), use regional compilation with dynamic shapes:**

```bash
python run_flux8.py \
  --prompt "A serene mountain landscape" \
  --compile --compile-regional --compile-dynamic \
  --width 1280 --height 720 \
  --json results.json
```

**Why this configuration?**
- **Regional compilation** is more robust on MPS backend (fewer graph breaks)
- **Dynamic shapes** prevents recompilation when generating different resolutions
- **Faster cold-start** compared to full compilation (~7-10x faster)
- **Better MPS compatibility** with smaller compilation units

### Compilation Options Explained

#### 1. Regional Compilation (Recommended for macOS)

Regional compilation targets repeated transformer blocks instead of the entire model:

```bash
--compile --compile-regional
```

**Benefits:**
- Compile time: ~10s instead of ~60s+ (on CUDA; MPS may vary)
- Same ~1.5x runtime speedup as full compilation
- More stable on MPS (macOS Metal backend)
- Better for iterative development

#### 2. Dynamic Shapes (Highly Recommended)

Prevents expensive recompilation when image dimensions change:

```bash
--compile --compile-dynamic
```

**Use when:**
- Generating images at multiple resolutions in the same session
- Experimenting with different aspect ratios
- Running batch jobs with varying dimensions

#### 3. Compilation Modes

Choose optimization strategy with `--compile-mode`:

```bash
# Balanced (default) - good for repeated generation
--compile --compile-mode reduce-overhead

# Maximum optimization - longer compile time, potentially better runtime
--compile --compile-mode max-autotune

# Standard compilation
--compile --compile-mode default
```

### macOS/MPS Considerations

**Important:** `torch.compile` on macOS with the MPS (Metal Performance Shaders) backend has **limited support** compared to CUDA:

⚠️ **Known Limitations:**
- Some operations may fall back to eager mode (no compilation)
- Graph breaks are more common than on CUDA
- Compilation may not provide speedups on all M-series chips
- First-run compile overhead can be significant

✅ **Recommended Approach:**
1. **Always use** `--compile-regional` instead of full compilation
2. **Always add** `--compile-dynamic` for flexibility
3. **Test first** - compile may or may not improve performance on your specific Mac
4. **Be patient** - first run includes compilation overhead

❌ **If compilation fails or errors occur:**
```bash
# Option 1: Run without compilation (rely on MPS native optimizations)
python run_flux8.py --prompt "..." --width 1280 --height 720

# Option 2: Use CPU (much slower, but always works)
# Not recommended - extremely slow
```

### Performance Expectations

#### On CUDA GPUs (NVIDIA RTX 5090, etc.)

| Configuration | Compile Time (Cold) | Runtime Speedup | Best For |
|--------------|---------------------|-----------------|----------|
| No compile | 0s | Baseline | Quick tests |
| Full compile | ~60s+ | ~1.5x faster | Production, single resolution |
| Regional compile | ~10s | ~1.5x faster | Development, multiple resolutions |
| Regional + dynamic | ~10s | ~1.5x faster | Maximum flexibility |

#### On Apple Silicon (M1/M2/M3/M4 Max/Ultra)

| Configuration | Notes |
|--------------|-------|
| No compile | Native MPS performance - already optimized |
| Regional + dynamic | **Recommended** - May provide 1.2-1.5x speedup, better compatibility |
| Full compile | Not recommended - Higher failure rate, longer compile time |

**Reality Check for macOS Users:**
- MPS backend is already highly optimized for Apple Silicon
- `torch.compile` gains on MPS are typically **smaller** than on CUDA
- Regional compilation is more about **compatibility** than dramatic speedups
- Your mileage may vary - test on your specific hardware

### Complete Examples

#### macOS Production (Recommended)
```bash
python run_flux8.py \
  --prompt "Cyberpunk city at night with neon lights" \
  --compile --compile-regional --compile-dynamic \
  --width 1280 --height 720 \
  --num_inference_steps 6 \
  --batch-size 10 \
  --sub-batch-size 1 \
  --json mac_benchmark.json \
  --output cyberpunk.png
```

#### CUDA Maximum Performance
```bash
python run_flux8.py \
  --prompt "Cyberpunk city at night with neon lights" \
  --compile --compile-mode max-autotune \
  --width 1280 --height 720 \
  --num_inference_steps 6 \
  --batch-size 10 \
  --sub-batch-size 1 \
  --json cuda_benchmark.json \
  --output cyberpunk.png
```

#### Multi-Resolution Workflow
```bash
# First run at 1280x720 (compilation happens here)
python run_flux8.py \
  --prompt "Mountain landscape" \
  --compile --compile-regional --compile-dynamic \
  --width 1280 --height 720 \
  --output mountain_hd.png

# Second run at 1024x1024 (no recompilation with --compile-dynamic)
python run_flux8.py \
  --prompt "Mountain landscape" \
  --compile --compile-regional --compile-dynamic \
  --width 1024 --height 1024 \
  --output mountain_square.png
```

### Troubleshooting Compilation

#### Compilation Takes Forever
- **Expected on first run** - Can take 10-60 seconds depending on configuration
- Use `--compile-regional` to reduce compile time by ~7x
- Consider if compilation is worth it for your use case

#### Compilation Fails on macOS
```bash
⚠️ WARNING: torch.compile on MPS (macOS) has limited support
```
This is normal. Options:
1. Try without compilation - MPS is already fast
2. Verify you're using PyTorch 2.4+ from Apple's PyPI
3. Check for PyTorch updates: `uv pip install --upgrade torch`

#### No Performance Improvement
Possible causes:
- MPS backend: Native optimizations may already be near-optimal
- Small batch sizes: Compilation overhead dominates
- Old PyTorch version: Update to 2.4+
- CPU fallback: Check that MPS is actually being used

#### Memory Increases After Compilation
- Expected behavior - compiled graphs use additional memory
- On memory-constrained systems, compilation may not be beneficial
- Try running without compilation if you hit OOM errors

## Output Files

### Generated Images

Images are saved with the following naming convention:
```
{base}_batch{N}_img{M}_seed{S}.{ext}
```

Example with `--output pink_elephant.png --batch-size 16`:
```
pink_elephant_batch1_img1_seed777.png
pink_elephant_batch2_img1_seed778.png
pink_elephant_batch3_img1_seed779.png
...
```

### JSON Performance Report (run_flux8.py)

The JSON report contains:

```json
{
  "system_info": {
    "platform": {...},
    "python": {...},
    "pytorch": {...},
    "cuda": {...},
    "packages": {...}
  },
  "generation_parameters": {
    "prompt": "...",
    "guidance_scale": 1.5,
    "num_inference_steps": 6,
    ...
  },
  "performance": {
    "model_loading_time": 1.12,
    "sub_batch_times": [...],
    "total_generation_time": 402.20,
    "average_time_per_image": 25.14,
    "images_per_second": 0.04
  },
  "memory_usage": {
    "gpu_memory": {...},
    "ram_usage": {...}
  }
}
```

## Model Information

- **Model**: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- **Type**: Text-to-image diffusion model
- **Precision**: bfloat16
- **Recommended Steps**: 4 (optimized for speed)
- **Guidance Scale**: 0.0 (schnell variant doesn't need guidance)

### Memory Requirements

Approximate peak VRAM usage (with CPU offloading):

| Resolution | VRAM (CUDA) | RAM |
|------------|-------------|-----|
| 1024×1024 | ~23 GB | ~15 GB |
| 1280×720 | ~23 GB | ~15 GB |
| 512×512 | ~20 GB | ~12 GB |

*Note: First run downloads ~35GB model files from Hugging Face*

## Performance Examples

### RTX 5090 (CUDA 12.8, Ubuntu)

```
Resolution: 1280×720
Batch size: 16 images
Model loading: 1.12s
Average per image: 25.14s
Peak VRAM: 23.1 GB
```

### Apple M-series (MPS, macOS)

Performance varies by chip generation and RAM configuration. The model will use unified memory.

## Architecture

### Device Selection

Scripts automatically detect and use the best available device:

1. **Apple Silicon MPS** - If available on macOS
2. **CUDA GPU** - If NVIDIA GPU is available
3. **CPU** - Fallback (very slow)

### Memory Management

**CUDA Mode**:
- Uses `enable_model_cpu_offload()` to reduce VRAM usage
- Keeps less-used model components in system RAM
- Enables generation on GPUs with limited VRAM

**MPS/CPU Mode**:
- Loads entire model to device with `.to(device)`
- Uses unified memory on Apple Silicon

### Batch Processing Strategy

To avoid out-of-memory errors:

1. Set `--batch-size` to total images needed
2. Set `--sub-batch-size` to 1 (or small value)
3. Each image uses a unique seed (starting from `--seed`)
4. Optional cooling periods with `--wait-between-sub-batches`

## Troubleshooting

### CUDA Out of Memory

```bash
# Use CPU offloading (enabled by default)
# Reduce resolution
python run_flux8.py --prompt "..." --width 512 --height 512

# Generate one at a time
python run_flux8.py --prompt "..." --batch-size 16 --sub-batch-size 1
```

### Slow Generation on CPU

CPU generation is very slow (10-100x slower than GPU). Consider:
- Using a system with CUDA or MPS support
- Reducing image resolution
- Reducing inference steps (minimum 4 for schnell)

### Model Download Issues

First run downloads ~35GB. Ensure:
- Stable internet connection
- Sufficient disk space (50GB free recommended)
- Hugging Face access (model is publicly available)

### Import Errors

```bash
# Reinstall dependencies
source .venv/bin/activate
uv pip install --force-reinstall torch diffusers transformers accelerate
```

## Dependencies

Core packages:
- **PyTorch** (`torch`) - Deep learning framework
- **Diffusers** (`diffusers`) - Hugging Face diffusion models
- **Transformers** (`transformers`) - Text encoding models
- **Accelerate** (`accelerate`) - Memory optimization
- **Pillow** (`pillow`) - Image handling
- **psutil** (`psutil`) - System monitoring (run_flux8.py)

See `requirements_*.txt` for complete platform-specific dependency lists.

## License

This code is provided as-is for research and educational purposes. The FLUX.1-schnell model has its own license terms - please refer to the [model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell) for details.

## Contributing

This is a benchmarking and testing repository. Feel free to:
- Add new script variations
- Improve performance monitoring
- Add support for other platforms
- Share benchmark results

## Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for the FLUX.1-schnell model
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- The PyTorch team for the ML framework
