
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

def setup_parser():
    parser = argparse.ArgumentParser(description="Convert IAM Handwriting Database to JSONL manifest")
    parser.add_argument("--iam-dir", type=str, required=True, help="Root directory of IAM dataset (containing lines.txt and lines/ folder)")
    parser.add_argument("--output", type=str, default="data/manifests/iam_train.jsonl", help="Output manifest path")
    parser.add_argument("--split-val", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--split-test", type=float, default=0.05, help="Test split ratio")
    return parser

def parse_lines_txt(lines_path):
    """
    Parse the lines.txt metadata file from IAM.
    Format: a01-000u-00 OK 154 408 768 27 51 AT A01-000u part of
    """
    samples = []
    print(f"Parsing {lines_path}...")
    
    with open(lines_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) < 9:
                continue
                
            line_id = parts[0]
            status = parts[1]
            # In lines.txt, the text is the last field. Words are separated by '|'.
            text_raw = parts[-1]
            text = text_raw.replace('|', ' ')
            
            # Filter out bad quality lines
            if status != 'ok':
                continue
                
            samples.append({
                'id': line_id,
                'text': text
            })
            
    return samples

def build_image_path(iam_root, line_id):
    """
    Construct path: root/lines/a01/a01-000u/a01-000u-00.png
    """
    # ID format: a01-000u-00
    parts = line_id.split('-')
    top_folder = parts[0]
    sub_folder = f"{parts[0]}-{parts[1]}"
    
    # Try different extensions
    rel_path = os.path.join('lines', top_folder, sub_folder, f"{line_id}.png")
    abs_path = os.path.join(iam_root, rel_path)
    
    if os.path.exists(abs_path):
        return abs_path
    
    return None

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    lines_txt = os.path.join(args.iam_dir, "lines.txt")
    if not os.path.exists(lines_txt):
        # Maybe inside ascii folder?
        lines_txt = os.path.join(args.iam_dir, "ascii", "lines.txt")
        
    if not os.path.exists(lines_txt):
        print(f"Error: Could not find lines.txt in {args.iam_dir}")
        return

    samples = parse_lines_txt(lines_txt)
    print(f"Found {len(samples)} valid samples in lines.txt")
    
    valid_records = []
    
    print("Verifying images...")
    for s in tqdm(samples):
        img_path = build_image_path(args.iam_dir, s['id'])
        if img_path:
            valid_records.append({
                "image_path": os.path.abspath(img_path),
                "ground_truth_text": s['text'],
                "mode": "text"
            })
            
    print(f"Matched {len(valid_records)} images.")
    
    # Split
    import random
    random.seed(42)
    random.shuffle(valid_records)
    
    n = len(valid_records)
    n_val = int(n * args.split_val)
    n_test = int(n * args.split_test)
    n_train = n - n_val - n_test
    
    train = valid_records[:n_train]
    val = valid_records[n_train:n_train+n_val]
    test = valid_records[n_train+n_val:]
    
    # Write
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    base, ext = os.path.splitext(args.output)
    
    def write_jsonl(data, path):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    write_jsonl(train, args.output)
    write_jsonl(val, f"{base.replace('_train', '')}_val{ext}")
    write_jsonl(test, f"{base.replace('_train', '')}_test{ext}")
    
    print(f"Saved: {len(train)} train, {len(val)} val, {len(test)} test")

if __name__ == "__main__":
    main()
