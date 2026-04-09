# SEEM API & Code Reference

This document provides a comprehensive guide to the SEEM inference API, allowing developers to integrate multi-modal segmentation into Python applications.

## 1. Core Model API

### `seem.inference.build_seem_model`

The factory function to instantiate the SEEM model architecture and load pre-trained weights.

```python
from seem.inference import build_seem_model

def build_seem_model(
    config_path: str,
    ckpt_path: str,
    device: str = "cuda"
) -> BaseModel
```

**Parameters:**
- `config_path` (str): Path to the SEEM YAML configuration file (e.g., `configs/seem_focall_lang.yaml`).
- `ckpt_path` (str): Path to the PyTorch checkpoint (`.pt`).
- `device` (str): Hardware device to load the model onto (`"cuda"` or `"cpu"`).

**Returns:**
- An initialized PyTorch `nn.Module` containing the SEEM architecture, set to `eval()` mode.

## 2. Inference Execution

The initialized model provides a unified `predict` interface.

### `model.predict`

Executes the forward pass using multi-modal prompts.

```python
def predict(
    self,
    image: torch.Tensor,
    prompts: dict
) -> dict
```

**Parameters:**
- `image` (torch.Tensor): The input image tensor. Must be preprocessed (RGB, normalized, shaped `(C, H, W)`).
- `prompts` (dict): A dictionary containing the multi-modal prompts. You can provide one or multiple keys simultaneously.
  - **Supported Keys:**
    - `"text"` (str or list of str): e.g., `"the red car"`.
    - `"point"` (list): A list of `[x, y]` coordinates. e.g., `[[250, 300]]`.
    - `"box"` (list): A bounding box `[x_min, y_min, x_max, y_max]`.
    - `"scribble"` (list of lists): Polygons defining a drawn line.
    - `"audio"` (str): Path to a `.wav` file containing spoken instructions.

**Returns:**
- `dict`: A dictionary containing the inference results.
  - `"mask"` (torch.Tensor): The generated binary or soft mask logits of shape `(1, H, W)`.
  - `"classes"` (list, optional): If panoptic segmentation is triggered, returns the semantic class labels.

## 3. Data Processing Utilities

To feed data into `model.predict`, you must use the provided data mappers.

### `seem.utils.image_processing.load_image`

```python
def load_image(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]
```
Reads an image from disk, converts to RGB, resizes according to the model's configuration constraints (usually maintaining aspect ratio), and normalizes it using ImageNet means/stds.
- Returns the prepared tensor and the original `(Height, Width)` for post-processing projection.

### `seem.utils.visualizer.Visualizer`

A utility class to draw the predicted masks onto the original image.

```python
from seem.utils.visualizer import Visualizer

vis = Visualizer(image_rgb)
vis.draw_binary_mask(mask_tensor, color="blue", alpha=0.5)
result_img = vis.get_image()
```

## 4. CLI Tools and Demos

### `demo/video_demo.py`

Processes a video file using referring expression segmentation (text prompt).

```bash
python demo/video_demo.py --video <input.mp4> --text "<description>" --output <out.mp4>
```

**Arguments:**
- `--video`: Input video path.
- `--text`: The referring expression (e.g., "the person wearing a hat").
- `--output`: Path to save the annotated video.
- `--cfg`: Path to model config.
- `--weight`: Path to model weights.

### Gradio Web UI (`assets/scripts/run_demo.sh`)

Launch the interactive local web application to test multi-modal inputs visually.

```bash
# Inside the run_demo.sh script, it executes:
python demo/app.py --conf_files configs/seem_focall_lang.yaml --ckpt seem_focall_v1.pt
```
This binds to `http://0.0.0.0:6090` and provides a UI for uploading images, clicking points, typing text, and uploading audio clips to instantly visualize SEEM's capabilities.
