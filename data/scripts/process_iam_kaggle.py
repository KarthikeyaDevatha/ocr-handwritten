
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

def setup_parser():
    parser = argparse.ArgumentParser(description="Convert Kaggle IAM Word Database to JSONL manifest by synthesizing lines")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Root directory containing words.txt and iam_words/words")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_lines_iam", help="Where to save synthesized line images")
    parser.add_argument("--manifest-output", type=str, default="data/manifests/iam_train.jsonl", help="Output manifest path")
    parser.add_argument("--check-words-new", action="store_true", help="Look for words_new.txt instead of words.txt")
    return parser

def parse_words_txt(txt_path):
    """
    Parse words.txt
    Format: a01-000u-00-00 ok 154 408 768 27 51 AT A
    """
    word_records = []
    print(f"Parsing {txt_path}...")
    
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            
            # parts[0]: id
            # parts[1]: status
            # parts[8:]: transcription (could be multiple if spaces escaped? usually just last)
            
            word_id = parts[0]
            status = parts[1]
            
            # Transcription is the last part usually, but let's be careful.
            # Official IAM uses last field.
            text = parts[-1]
            
            # Skip errors?
            if status == 'err':
                continue
                
            word_records.append({
                'id': word_id,
                'text': text,
                'status': status
            })
            
    return word_records

def get_word_image_path(dataset_dir, word_id):
    """
    Path: dataset_dir/iam_words/words/a01/a01-000u/a01-000u-00-00.png
    """
    parts = word_id.split('-')
    folder1 = parts[0]
    folder2 = f"{parts[0]}-{parts[1]}"
    
    # Try different structures
    # Structure 1: iam_words/words/...
    path1 = os.path.join(dataset_dir, "iam_words", "words", folder1, folder2, f"{word_id}.png")
    if os.path.exists(path1): return path1
    
    # Structure 2: words/... (flat inside dataset dir?)
    path2 = os.path.join(dataset_dir, "words", folder1, folder2, f"{word_id}.png")
    if os.path.exists(path2): return path2
    
    return None

def synthesize_lines(word_records, dataset_dir, output_dir):
    """
    Group words by line ID and concatenate images.
    """
    # Group by line ID (first 3 components: a01-000u-00)
    lines = {}
    for w in word_records:
        # ID: a01-000u-00-00
        parts = w['id'].split('-')
        if len(parts) < 4: continue
        
        line_id = "-".join(parts[:3])
        if line_id not in lines:
            lines[line_id] = []
        lines[line_id].append(w)
        
    print(f"grouped into {len(lines)} lines")
    
    os.makedirs(output_dir, exist_ok=True)
    manifest_data = []
    
    success_count = 0
    
    for line_id, words in tqdm(lines.items()):
        # Sort words by index (last part)
        # However, indices are strings '00', '01'. Sorting works.
        words.sort(key=lambda x: x['id'])
        
        # Load images
        images = []
        texts = []
        
        for w in words:
            img_path = get_word_image_path(dataset_dir, w['id'])
            if img_path:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    texts.append(w['text'])
                except Exception:
                    pass
        
        if not images:
            continue
            
        # Concatenate
        # Calculate dimensions
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Add random spacing (simulate handwriting spacing)
        # Average space width... say 30px +/- 10
        space_width = 30
        total_intervals = max(0, len(images) - 1)
        # We will add variable space between words
        
        # Create canvas
        # Height: max_height + padding
        # Width: total_width + (total_intervals * space_width)
        
        # Pre-calc spaces
        spaces = [random.randint(20, 50) for _ in range(total_intervals)]
        final_width = total_width + sum(spaces)
        
        line_img = Image.new('RGB', (final_width, max_height), 'white')
        
        current_x = 0
        for i, img in enumerate(images):
            # Center vertically? or Align top? Or align baseline?
            # Align center usually safer for generic fallback
            y_offset = (max_height - img.height) // 2
            line_img.paste(img, (current_x, y_offset))
            current_x += img.width
            if i < len(spaces):
                current_x += spaces[i]
                
        # Save
        out_filename = f"{line_id}.png"
        out_path = os.path.join(output_dir, out_filename)
        line_img.save(out_path)
        
        # Text
        full_text = " ".join(texts)
        
        manifest_data.append({
            "image_path": os.path.abspath(out_path),
            "ground_truth_text": full_text,
            "mode": "text"
        })
        success_count += 1
        
    return manifest_data

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    txt_filename = "words_new.txt" if args.check_words_new else "words.txt"
    txt_path = os.path.join(args.dataset_dir, txt_filename)
    
    if not os.path.exists(txt_path):
        # Check inside iam_words?
        txt_path_inner = os.path.join(args.dataset_dir, "iam_words", txt_filename)
        if os.path.exists(txt_path_inner):
            txt_path = txt_path_inner
        else:
            # Try finding any txt
            import glob
            txts = glob.glob(os.path.join(args.dataset_dir, "*.txt"))
            if txts:
                txt_path = txts[0]
                print(f"Warning: Using found text file {txt_path}")
            else:
                print(f"Error: Could not find {txt_filename} in {args.dataset_dir}")
                # Try words.txt if looking for words_new
                if args.check_words_new:
                     txt_path = os.path.join(args.dataset_dir, "words.txt")
                     if not os.path.exists(txt_path): return
                else: return

    records = parse_words_txt(txt_path)
    print(f"Found {len(records)} word records.")
    
    manifest = synthesize_lines(records, args.dataset_dir, args.output_dir)
    print(f"Synthesized {len(manifest)} lines.")
    
    # Split
    random.seed(42)
    random.shuffle(manifest)
    
    n = len(manifest)
    n_val = int(n * 0.05)
    n_test = int(n * 0.05)
    n_train = n - n_val - n_test
    
    train = manifest[:n_train]
    val = manifest[n_train:n_train+n_val]
    test = manifest[n_train+n_val:]
    
    base, ext = os.path.splitext(args.manifest_output)
    
    def write_jsonl(data, path):
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    write_jsonl(train, args.manifest_output)
    write_jsonl(val, f"{base.replace('_train', '')}_val{ext}")
    write_jsonl(test, f"{base.replace('_train', '')}_test{ext}")
    
    print("Saved manifests.")

if __name__ == "__main__":
    main()
