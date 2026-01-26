"""
Token Edit Distance computation for math OCR evaluation.
Evaluates LaTeX expression recognition quality.
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Tuple

from tqdm import tqdm


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize LaTeX expression into atomic tokens.
    
    Args:
        latex: LaTeX string
    
    Returns:
        List of tokens
    """
    latex = latex.strip()
    
    # Remove $ delimiters
    if latex.startswith('$'):
        latex = latex.lstrip('$')
    if latex.endswith('$'):
        latex = latex.rstrip('$')
    
    tokens = []
    i = 0
    
    while i < len(latex):
        # Skip whitespace
        if latex[i].isspace():
            i += 1
            continue
        
        # LaTeX command
        if latex[i] == '\\':
            j = i + 1
            while j < len(latex) and latex[j].isalpha():
                j += 1
            if j == i + 1:
                # Single character after backslash
                if j < len(latex):
                    tokens.append(latex[i:j+1])
                    i = j + 1
                else:
                    tokens.append('\\')
                    i = j
            else:
                tokens.append(latex[i:j])
                i = j
        
        # Braces, subscript, superscript
        elif latex[i] in '{}^_':
            tokens.append(latex[i])
            i += 1
        
        # Numbers
        elif latex[i].isdigit():
            j = i
            while j < len(latex) and (latex[j].isdigit() or latex[j] == '.'):
                j += 1
            tokens.append(latex[i:j])
            i = j
        
        # Letters
        elif latex[i].isalpha():
            tokens.append(latex[i])
            i += 1
        
        # Other characters
        else:
            tokens.append(latex[i])
            i += 1
    
    return tokens


def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two token sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
    
    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    m, n = len(seq1), len(seq2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[m][n]


def token_edit_distance(prediction: str, reference: str) -> Tuple[int, int]:
    """
    Compute token-level edit distance for LaTeX expressions.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Tuple of (edit_distance, reference_length)
    """
    pred_tokens = tokenize_latex(prediction)
    ref_tokens = tokenize_latex(reference)
    
    dist = edit_distance(pred_tokens, ref_tokens)
    
    return dist, len(ref_tokens)


def normalized_edit_distance(prediction: str, reference: str) -> float:
    """
    Compute normalized token edit distance.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Normalized distance (0.0 to 1.0+)
    """
    dist, ref_len = token_edit_distance(prediction, reference)
    
    if ref_len == 0:
        return 0.0 if dist == 0 else 1.0
    
    return dist / ref_len


def expression_exact_match(prediction: str, reference: str) -> bool:
    """
    Check if expressions match exactly (ignoring whitespace).
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        True if exact match
    """
    # Normalize whitespace
    pred_norm = re.sub(r'\s+', '', prediction)
    ref_norm = re.sub(r'\s+', '', reference)
    
    return pred_norm == ref_norm


def evaluate_math_predictions(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Evaluate math OCR predictions.
    
    Args:
        predictions: List of predicted LaTeX
        references: List of ground truth LaTeX
    
    Returns:
        Evaluation metrics
    """
    assert len(predictions) == len(references), "Mismatched lengths"
    
    n = len(predictions)
    
    total_dist = 0
    total_len = 0
    exact_matches = 0
    
    distances = []
    
    for pred, ref in zip(predictions, references):
        dist, ref_len = token_edit_distance(pred, ref)
        
        total_dist += dist
        total_len += ref_len
        distances.append(normalized_edit_distance(pred, ref))
        
        if expression_exact_match(pred, ref):
            exact_matches += 1
    
    return {
        "token_edit_distance_mean": sum(distances) / n if n > 0 else 0,
        "token_edit_distance_median": sorted(distances)[n // 2] if n > 0 else 0,
        "total_edit_operations": total_dist,
        "total_reference_tokens": total_len,
        "overall_token_accuracy": 1 - (total_dist / total_len) if total_len > 0 else 0,
        "expression_exact_match_rate": exact_matches / n if n > 0 else 0,
        "num_samples": n
    }


def run_evaluation(
    manifest_path: str,
    model_dir: str,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Run full math OCR evaluation.
    
    Args:
        manifest_path: Path to test manifest
        model_dir: Path to math OCR model directory
        device: Compute device
    
    Returns:
        Evaluation metrics
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference.ocr_math import create_math_ocr
    from inference.preprocess import preprocess_for_ocr
    
    # Load samples
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            if sample.get('mode') in ['math', 'mixed']:
                samples.append(sample)
    
    print(f"Evaluating on {len(samples)} math samples...")
    
    # Initialize OCR
    ocr = create_math_ocr(model_dir=model_dir, device=device)
    
    predictions = []
    references = []
    
    for sample in tqdm(samples, desc="Running Math OCR"):
        image_path = sample['image_path']
        ground_truth = sample['ground_truth_text']
        
        try:
            image = preprocess_for_ocr(image_path)
            result = ocr.recognize(image)
            predictions.append(result.latex)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions.append("")
        
        references.append(ground_truth)
    
    # Compute metrics
    metrics = evaluate_math_predictions(predictions, references)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute token edit distance for math OCR")
    parser.add_argument("--manifest", type=str, required=True, help="Path to test manifest")
    parser.add_argument("--model-dir", type=str, default="models/trocr_math", help="Math OCR model directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    metrics = run_evaluation(
        manifest_path=args.manifest,
        model_dir=args.model_dir,
        device=args.device
    )
    
    print("\n=== Math OCR Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
