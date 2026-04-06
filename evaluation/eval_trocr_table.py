import os
import sys
import json
import time

# Add root project dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.ocr_text import create_text_ocr
from inference.preprocess import preprocess_for_ocr
from evaluation.compute_cer import compute_cer, compute_wer

def evaluate():
    manifest_path = "data/manifests/train.jsonl"
    model_dir = "models/trocr_text"
    
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
            
    samples = samples[:10]
                
    print(f"Loaded {len(samples)} text samples from {manifest_path}")
    
    ocr = create_text_ocr(model_dir=model_dir, device="cpu")
    
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    total_time = 0.0
    
    for sample in samples:
        image_path = sample['image_path']
        gt = sample['ground_truth_text']
        
        try:
            # Load and preprocess image
            image = preprocess_for_ocr(image_path)
            
            # Predict
            start_t = time.time()
            result = ocr.recognize(image)
            end_t = time.time()
            
            pred = result.text
            
            # Compute metrics
            cer = compute_cer(pred, gt)
            wer = compute_wer(pred, gt)
            
            total_cer += cer
            total_wer += wer
            if pred.strip() == gt.strip():
                exact_matches += 1
                
            total_time += (end_t - start_t)
            
            print(f"[{image_path}] GT: '{gt}' | PRED: '{pred}' | CER: {cer:.2f}")
            
        except Exception as e:
            print(f"Error on {image_path}: {e}")
            
    n = len(samples)
    if n > 0:
        print("\n--- TrOCR PROJECT METRICS ---")
        print(f"CER: {total_cer / n:.4f}")
        print(f"WER: {total_wer / n:.4f}")
        print(f"Recognition Accuracy (%): {(exact_matches / n) * 100:.2f}%")
        print(f"Processing Time (sec/page): {total_time / n:.4f}")

if __name__ == '__main__':
    evaluate()
