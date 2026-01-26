# Handwritten OCR System

Local-only, edge-deployable OCR system for recognizing mixed handwritten English text and mathematical expressions (LaTeX) from images.

## Features

- **Modular Pipeline**: Segmentation → Classification → OCR → Reconstruction
- **Dual Recognition**: Separate models for text (TrOCR) and math (MFR)
- **Layout Detection**: YOLOv8-Nano for detecting text lines and math formulas
- **ONNX Runtime**: INT8 quantized models for fast CPU inference
- **Confidence Routing**: Automatic rerouting based on prediction confidence
- **Markdown + LaTeX Output**: Clean, renderable document output

## Requirements

- Python 3.9+
- PyTorch 2.1+
- ONNX Runtime 1.17+

## Installation

```bash
# Clone the repository
cd "OCR PROJECT"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data

```bash
# Generate synthetic training data
python data/synthesis_script.py --num-samples 1000

# Generate a sample page for testing
python data/synthesis_script.py --generate-page
```

### 2. Adapt Tokenizer (Optional)

```bash
python training/tokenizer_utils.py --output models/trocr_text/tokenizer
```

### 3. Train Models

#### Text OCR (TrOCR)

```bash
python training/train_trocr_text.py \
  --manifest data/manifests/train.jsonl \
  --val-manifest data/manifests/val.jsonl \
  --epochs 10 \
  --batch_size 8 \
  --lr 5e-5 \
  --output checkpoints/trocr_text \
  --adapt-tokenizer
```

#### Math OCR (MFR)

```bash
python training/train_mfr_math.py \
  --manifest data/manifests/train.jsonl \
  --val-manifest data/manifests/val.jsonl \
  --epochs 8 \
  --batch_size 4 \
  --output checkpoints/trocr_math \
  --adapt-latex
```

#### Layout Detection (YOLOv8)

```bash
# Generate layout training data
python training/train_yolo.py --action generate --data-dir data/yolo_layout

# Train YOLO
python training/train_yolo.py --action train --data-dir data/yolo_layout --epochs 50

# Export to ONNX
python training/train_yolo.py --action export --export-path models/yolo/yolov8n-layout.onnx
```

### 4. Export to ONNX

```bash
# Text OCR
optimum-cli export onnx \
  --model checkpoints/trocr_text/final \
  --task image-to-text-with-past \
  models/trocr_text/

# Math OCR
optimum-cli export onnx \
  --model checkpoints/trocr_math/final \
  --task image-to-text-with-past \
  models/trocr_math/
```

### 5. Run Inference

```bash
# Process a single image
python inference/pipeline.py \
  --image samples/page.png \
  --output outputs/demo_result.md \
  --debug
```

### 6. Run Streamlit Demo

```bash
streamlit run app/app.py
```

## Project Structure

```
project_root/
├── data/
│   ├── raw/                  # IAM, MathWriting, IBEM datasets
│   ├── synthetic/            # Generated synthetic data
│   ├── manifests/            # JSONL data manifests
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── synthesis_script.py   # Data generation script
├── models/
│   ├── yolo/
│   │   └── yolov8n-layout.onnx
│   ├── trocr_text/
│   │   ├── encoder_model.onnx
│   │   ├── decoder_model.onnx
│   │   └── tokenizer/
│   └── trocr_math/
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       └── tokenizer/
├── training/
│   ├── train_trocr_text.py   # Text OCR training
│   ├── train_mfr_math.py     # Math OCR training
│   ├── train_yolo.py         # Layout model training
│   └── tokenizer_utils.py    # Tokenizer adaptation
├── inference/
│   ├── preprocess.py         # Image preprocessing
│   ├── layout.py             # Layout detection
│   ├── ocr_text.py           # Text recognition
│   ├── ocr_math.py           # Math recognition
│   ├── reconstruct.py        # Document reconstruction
│   └── pipeline.py           # Full inference pipeline
├── evaluation/
│   ├── compute_cer.py        # Character Error Rate
│   ├── compute_token_distance.py  # Token Edit Distance
│   └── compare_checkpoints.py     # Checkpoint comparison
├── app/
│   └── app.py                # Streamlit demo
├── outputs/                  # Generated outputs
├── samples/                  # Sample images
├── requirements.txt
└── README.md
```

## Evaluation

### Text OCR (CER/WER)

```bash
python evaluation/compute_cer.py \
  --manifest data/manifests/test.jsonl \
  --model-dir models/trocr_text
```

### Math OCR (Token Edit Distance)

```bash
python evaluation/compute_token_distance.py \
  --manifest data/manifests/test.jsonl \
  --model-dir models/trocr_math
```

### Compare Checkpoints

```bash
python evaluation/compare_checkpoints.py \
  --checkpoint-dir checkpoints/trocr_text \
  --manifest data/manifests/val.jsonl \
  --model-type text
```

## Performance Targets

| Metric | Target | Device |
|--------|--------|--------|
| Per-line inference | <1s | CPU (Intel i5/i7) |
| Per-line inference | <200ms | GPU (CUDA) |
| Full page (10-15 lines) | 5-10s | CPU |
| Model size (ONNX INT8) | <100MB | - |

## API Usage

```python
from inference.pipeline import OCRPipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig(device="cpu")
pipeline = OCRPipeline(config)

# Process image
result = pipeline.process("path/to/image.png")

# Access results
print(result.markdown)
print(f"Processing time: {result.total_time:.2f}s")
print(f"Detected: {result.num_text_regions} text, {result.num_math_regions} math")
```

## Manifest Format

JSONL format for training data:

```json
{"image_path": "data/synthetic/img_000123.png", "ground_truth_text": "Calculate the integral $\\int x^2 dx$", "mode": "mixed"}
```

Modes: `text`, `math`, `mixed`

## Known Limitations

- Requires horizontal text (rotation >15° may affect accuracy)
- Best results with high-quality scans (300+ DPI)
- Complex multi-line equations may require manual correction

## License

MIT License
