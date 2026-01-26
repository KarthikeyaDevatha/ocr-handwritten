"""
YOLOv8 Layout Detection Training Script.
Trains YOLOv8-Nano for detecting text lines and math formulas.
"""

import os
import sys
import json
import argparse
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw

# Fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def create_yolo_dataset_yaml(output_dir: str, num_classes: int = 2) -> str:
    """Create YOLO dataset configuration YAML."""
    yaml_content = f"""# IBEM + Synthetic Layout Detection Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: text_line
  1: math_formula
"""
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def create_synthetic_layout_data(
    output_dir: str,
    num_train: int = 500,
    num_val: int = 100,
    num_test: int = 100,
    image_size: tuple = (640, 640)
):
    """Generate synthetic layout detection training data."""
    
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "test"), exist_ok=True)
    
    splits = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test)
    ]
    
    for split_name, num_samples in splits:
        print(f"Generating {num_samples} {split_name} samples...")
        
        for i in range(num_samples):
            # Create blank page
            img = Image.new('RGB', image_size, color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Generate random layout elements
            labels = []
            y = random.randint(20, 60)
            
            while y < image_size[1] - 80:
                # Decide element type (0 = text, 1 = math)
                elem_type = random.choice([0, 0, 0, 1])  # 75% text, 25% math
                
                # Random dimensions
                x = random.randint(20, 60)
                width = random.randint(200, image_size[0] - 100)
                height = random.randint(30, 60) if elem_type == 0 else random.randint(40, 100)
                
                # Draw rectangle to simulate content
                color = (random.randint(180, 220),) * 3
                draw.rectangle([x, y, x + width, y + height], fill=color)
                
                # Add some "text" lines
                for line_y in range(y + 5, y + height - 5, 15):
                    line_width = random.randint(width // 2, width - 10)
                    draw.line([(x + 5, line_y), (x + 5 + line_width, line_y)], 
                             fill=(random.randint(0, 50),) * 3, width=2)
                
                # Calculate YOLO format label (normalized center x, y, width, height)
                cx = (x + width / 2) / image_size[0]
                cy = (y + height / 2) / image_size[1]
                w = width / image_size[0]
                h = height / image_size[1]
                
                labels.append(f"{elem_type} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
                y += height + random.randint(20, 50)
            
            # Add noise
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy)
            
            # Save image and labels
            img_path = os.path.join(output_dir, "images", split_name, f"{i:06d}.jpg")
            label_path = os.path.join(output_dir, "labels", split_name, f"{i:06d}.txt")
            
            img.save(img_path, quality=95)
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
    
    return create_yolo_dataset_yaml(output_dir)


def train_yolo(
    dataset_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    output_dir: str = "checkpoints/yolo",
    device: str = ""
):
    """Train YOLOv8 model for layout detection."""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    print(f"Loading base model: {model}")
    yolo = YOLO(model)
    
    # Train
    print(f"Starting training for {epochs} epochs...")
    results = yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="layout",
        device=device if device else None,
        seed=SEED,
        batch=16,
        patience=10,
        save=True,
        save_period=5,
        val=True,
        plots=True,
    )
    
    return results


def export_yolo_onnx(
    model_path: str,
    output_path: str = "models/yolo/yolov8n-layout.onnx",
    imgsz: int = 640,
    simplify: bool = True,
    dynamic: bool = True
):
    """Export trained YOLO model to ONNX format."""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting {model_path} to ONNX...")
    yolo = YOLO(model_path)
    
    yolo.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
    )
    
    # Move exported file to desired location
    exported_path = model_path.replace(".pt", ".onnx")
    if os.path.exists(exported_path) and exported_path != output_path:
        shutil.copy(exported_path, output_path)
        print(f"Exported to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for layout detection")
    parser.add_argument("--action", choices=["generate", "train", "export"], required=True)
    parser.add_argument("--data-dir", type=str, default="data/yolo_layout")
    parser.add_argument("--dataset-yaml", type=str, help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default="checkpoints/yolo")
    parser.add_argument("--export-path", type=str, default="models/yolo/yolov8n-layout.onnx")
    parser.add_argument("--trained-model", type=str, help="Path to trained .pt for export")
    parser.add_argument("--device", type=str, default="", help="cuda device or cpu")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.action == "generate":
        print("Generating synthetic layout dataset...")
        yaml_path = create_synthetic_layout_data(
            args.data_dir,
            num_train=args.num_train,
            num_val=args.num_val
        )
        print(f"Dataset created at {args.data_dir}")
        print(f"YAML config: {yaml_path}")
    
    elif args.action == "train":
        yaml_path = args.dataset_yaml or os.path.join(args.data_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print(f"Dataset YAML not found: {yaml_path}")
            print("Run with --action generate first")
            sys.exit(1)
        
        train_yolo(
            yaml_path,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            output_dir=args.output,
            device=args.device
        )
    
    elif args.action == "export":
        model_path = args.trained_model
        if not model_path:
            # Try to find best.pt in default location
            model_path = os.path.join(args.output, "layout", "weights", "best.pt")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            sys.exit(1)
        
        export_yolo_onnx(
            model_path,
            output_path=args.export_path,
            imgsz=args.imgsz
        )


if __name__ == "__main__":
    main()
