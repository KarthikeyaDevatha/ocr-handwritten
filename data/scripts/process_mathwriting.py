
import os
import json
import argparse
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from multiprocessing import Pool, cpu_count
import functools

def setup_parser():
    parser = argparse.ArgumentParser(description="Convert Google MathWriting (InkML) to images and JSONL manifest")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .inkml files")
    parser.add_argument("--output-dir", type=str, default="data/processed/mathwriting", help="Where to save rendered images")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests", help="Where to save jsonl files")
    parser.add_argument("--img-size", type=int, default=256, help="Output image height (width auto)")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    return parser

def parse_inkml(file_path):
    """
    Parse an InkML file to extract traces and ground truth LaTeX.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Namespace map
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        # Extract Annotation (LaTeX)
        latex = None
        for annotation in root.findall('ink:annotation', ns):
            if annotation.get('type') == 'truth':
                latex = annotation.text
                break
        
        if not latex:
            return None, None
            
        # Extract Traces
        traces = []
        for trace in root.findall('ink:trace', ns):
            # Text is "x1 y1, x2 y2, ..."
            coords_text = trace.text.strip()
            # Split by comma first, then space
            points = []
            for pair in coords_text.split(','):
                pair = pair.strip()
                if not pair: continue
                try:
                    x, y = map(float, pair.split())
                    points.append((x, y))
                except ValueError:
                    continue
            if points:
                traces.append(points)
                
        return latex, traces
        
    except Exception as e:
        # print(f"Error parsing {file_path}: {e}")
        return None, None

def render_traces(traces, target_height=256):
    """
    Render traces to a PIL Image.
    """
    if not traces:
        return None
        
    # Find bounding box
    all_x = [p[0] for t in traces for p in t]
    all_y = [p[1] for t in traces for p in t]
    
    if not all_x or not all_y:
        return None
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Add padding
    padding = 20
    
    # Scale to target height
    scale = (target_height - 2*padding) / max(height, 1)
    
    new_width = int(width * scale + 2*padding)
    new_height = target_height # int(height * scale + 2*padding)
    
    # Create image
    img = Image.new('RGB', (new_width, new_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw traces
    for trace in traces:
        # Transform points
        points = []
        for x, y in trace:
            new_x = (x - min_x) * scale + padding
            new_y = (y - min_y) * scale + padding
            points.append((new_x, new_y))
            
        if len(points) > 1:
            draw.line(points, fill='black', width=3)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse([x-2, y-2, x+2, y+2], fill='black')
            
    return img

def process_file_item(file_path, output_dir, img_size):
    """
    Worker function to process a single file.
    Returns dict record or None.
    """
    latex, traces = parse_inkml(file_path)
    if not latex or not traces:
        return None
        
    # Create output filename
    basename = os.path.basename(file_path).replace('.inkml', '.png')
    # Hash path to avoid collisions if flattened? 
    # MathWriting usually has unique filenames.
    
    out_path = os.path.join(output_dir, basename)
    
    try:
        img = render_traces(traces, target_height=img_size)
        if img:
            img.save(out_path)
            return {
                "image_path": os.path.abspath(out_path),
                "ground_truth_text": latex,
                "mode": "math"
            }
    except Exception as e:
        return None
        
    return None

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.input_dir, "**/*.inkml"), recursive=True)
    if args.limit:
        files = files[:args.limit]
        
    print(f"Found {len(files)} files. Processing with {args.num_workers} workers...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)
    
    process_func = functools.partial(process_file_item, output_dir=args.output_dir, img_size=args.img_size)
    
    valid_records = []
    
    with Pool(args.num_workers) as p:
        for result in tqdm(p.imap_unordered(process_func, files), total=len(files)):
            if result:
                valid_records.append(result)
                
    print(f"Successfully processed {len(valid_records)} samples.")
    
    # Split 90/5/5
    import random
    random.seed(42)
    random.shuffle(valid_records)
    
    n = len(valid_records)
    n_val = int(n * 0.05)
    n_test = int(n * 0.05)
    n_train = n - n_val - n_test
    
    train = valid_records[:n_train]
    val = valid_records[n_train:n_train+n_val]
    test = valid_records[n_train+n_val:]
    
    def write_jsonl(data, name):
        path = os.path.join(args.manifest_dir, name)
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    write_jsonl(train, "mathwriting_train.jsonl")
    write_jsonl(val, "mathwriting_val.jsonl")
    write_jsonl(test, "mathwriting_test.jsonl")
    
    print("Manifests saved to", args.manifest_dir)

if __name__ == "__main__":
    main()
