
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob

def setup_parser():
    parser = argparse.ArgumentParser(description="Process Custom Drive Dataset (JSON + Images) into Line Crops")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .json and .jpg files (e.g. data/raw/custom_drive/extracted/train)")
    parser.add_argument("--output-dir", type=str, default="data/processed/custom_lines", help="Where to save cropped line images")
    parser.add_argument("--manifest-output", type=str, default="data/manifests/custom_train.jsonl", help="Output manifest path")
    parser.add_argument("--padding", type=int, default=10, help="Padding around line bounding box")
    return parser

def parse_polygons_to_bbox(words):
    """
    Compute bounding box for a list of words.
    Each word has "polygon": {"x0":..., "y0":..., "x1":...} or similar.
    Wait, the sample schema was: {"x0": 85, "y0": 132, "x1": 197, "y1": 128, ...}
    It seems to have x0,y0 to x3,y3 (4 points).
    """
    all_x = []
    all_y = []
    
    for w in words:
        poly = w.get('polygon', {})
        # extracting all keys x0..x3, y0..y3
        # Use simple iteration or known keys
        for key, val in poly.items():
            if key.startswith('x'):
                all_x.append(val)
            elif key.startswith('y'):
                all_y.append(val)
                
    if not all_x or not all_y:
        return None
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    return (min_x, min_y, max_x, max_y)

def process_file_pair(json_path, args):
    """
    Process a single JSON/Image pair.
    Returns list of manifest records.
    """
    # Find corresponding image
    base = os.path.splitext(json_path)[0]
    img_path = base + ".jpg"
    
    if not os.path.exists(img_path):
        # Try png?
        if os.path.exists(base + ".png"):
            img_path = base + ".png"
        else:
            # print(f"Warning: Image not found for {json_path}")
            return []
            
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Group by line_idx
        lines = {}
        for item in data:
            idx = item.get('line_idx')
            if idx is None: continue
            
            if idx not in lines:
                lines[idx] = []
            lines[idx].append(item)
            
        # Helper for sorting words left-to-right? 
        # Usually checking mean x centroid is good.
        
        records = []
        try:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return []
            
        for idx, words in lines.items():
            # Sort words by min x (x0) roughly to ensure correct text order
            # The JSON list order might be correct, but let's be safe.
            # Using x0 of polygon
            words.sort(key=lambda w: w['polygon'].get('x0', 0))
            
            bbox = parse_polygons_to_bbox(words)
            if not bbox: continue
            
            min_x, min_y, max_x, max_y = bbox
            
            # Add padding
            min_x = max(0, min_x - args.padding)
            min_y = max(0, min_y - args.padding)
            max_x = min(w, max_x + args.padding)
            max_y = min(h, max_y + args.padding)
            
            # Crop
            try:
                crop = image.crop((min_x, min_y, max_x, max_y))
                
                # Construct text
                line_text = " ".join([w['text'] for w in words])
                
                # Save
                filename = f"{os.path.basename(base)}_line{idx}.jpg"
                save_path = os.path.join(args.output_dir, filename)
                crop.save(save_path)
                
                records.append({
                    "image_path": os.path.abspath(save_path),
                    "ground_truth_text": line_text,
                    "mode": "text"
                })
            except Exception as e:
                pass
                
        return records
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_output), exist_ok=True)
    
    # Find all json files
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {args.input_dir}")
    
    all_records = []
    
    for json_file in tqdm(json_files):
        records = process_file_pair(json_file, args)
        all_records.extend(records)
        
    print(f"Processed {len(all_records)} line samples.")
    
    # Split
    import random
    random.seed(42)
    random.shuffle(all_records)
    
    val_ratio = 0.1
    n_val = int(len(all_records) * val_ratio)
    train_data = all_records[n_val:]
    val_data = all_records[:n_val]
    
    # Write manifests
    with open(args.manifest_output, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
            
    val_path = args.manifest_output.replace('train', 'val')
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved {len(train_data)} train and {len(val_data)} val samples.")

if __name__ == "__main__":
    main()
