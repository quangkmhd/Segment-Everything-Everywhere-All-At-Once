# Configuration Guide for SEEM

SEEM uses a hierarchical configuration system driven by YAML files. This document explains how to configure the model architecture, inference parameters, and environment settings.

## 1. YAML Configuration Files (`configs/`)

The core architecture and hyperparameters are defined in YAML files. These files dictate which backbone to load and what features to enable.

### Standard Configuration Profiles

| Config File | Backbone | Description | Use Case |
|-------------|----------|-------------|----------|
| `seem_focalt_lang.yaml` | FocalNet-Tiny | Lightweight model, fast inference. | Edge deployment, local testing on limited VRAM (< 10GB). |
| `seem_focall_lang.yaml` | FocalNet-Large | Heavy model, maximum accuracy. | High-end GPUs (24GB+ VRAM), production deployments. |

### Key YAML Hyperparameters

If you open `configs/seem_focall_lang.yaml`, you will see several blocks:

#### `MODEL.ENCODER`
Defines the vision backbone.
- `TYPE`: e.g., `"focal"`
- `DIM_MODEL`: Base channel dimension.

#### `MODEL.DECODER`
Defines the X-Decoder architecture.
- `HIDDEN_DIM`: (Default: `512`) The dimension of the query embeddings.
- `NUM_OBJECT_QUERIES`: (Default: `100`) The maximum number of distinct objects the model can track or predict simultaneously.

#### `TEST`
Inference-specific configurations.
- `TEST.MIN_SIZE_TEST`: (Default: `800`) The shorter edge of the image is resized to this pixel dimension before inference. 
  - *Tuning*: If you hit OOM errors on large images, reduce this to `512` or `480`. The segmentation boundaries might become slightly less crisp, but VRAM usage will plummet.
- `TEST.MAX_SIZE_TEST`: (Default: `1333`) The maximum allowed pixel dimension for the longer edge.

## 2. Environment Variables & Setup

### CUDA Custom Operators
SEEM relies on a custom implementation of Multi-Scale Deformable Attention. 
- You MUST compile this before running the code. The compilation respects the standard PyTorch CUDA environment variables.
- Ensure `CUDA_HOME` is set to your CUDA toolkit installation path (e.g., `/usr/local/cuda-11.8`) before running `python setup.py build develop` in the `modeling/pixel_decoder/ops` directory.
- `MAX_JOBS`: Set `export MAX_JOBS=4` during compilation to speed up the C++ build process.

### HuggingFace Hub (Optional)
If you are using the automated weight download scripts or loading the text encoder via transformers:
- `HF_HOME`: Dictates where downloaded huggingface models (like the CLIP text encoder) are cached. By default, this is `~/.cache/huggingface`.

## 3. Model Weights (Checkpoints)

You must ensure the downloaded `.pt` file matches the YAML configuration.
- Tiny config -> `seem_focalt_v1.pt`
- Large config -> `seem_focall_v1.pt`

Mismatching these will result in massive `RuntimeError: size mismatch` exceptions during the `torch.load()` step because the tensor dimensions in the checkpoint will not match the initialized PyTorch module graph.

## 4. Docker Configuration

If running via the provided Dockerfile:
- The `docker-compose.yml` or `docker run` command MUST include the `--gpus all` flag. SEEM operations (especially the custom deformable attention kernels) are not implemented for CPU execution and will crash if a CUDA device is not mapped into the container.
- Port mapping: The default Gradio demo binds to `6090`. Ensure `-p 6090:6090` is exposed.
