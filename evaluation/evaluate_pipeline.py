"""
Evaluation Harness for OCR Pipeline.
Computes CER, WER, LaTeX Exact Match, and BLEU metrics.
Supports A/B comparison between pipeline configurations.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EvalSample:
    """A single evaluation sample."""
    image_path: str
    ground_truth_text: str = ""
    ground_truth_latex: str = ""
    category: str = "general"  # "simple_math", "complex_math", "handwritten", "printed", "table"


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    image_path: str
    predicted_text: str
    ground_truth_text: str
    cer: float = 0.0
    wer: float = 0.0
    latex_exact_match: bool = False
    latex_edit_distance: float = 0.0
    latency: float = 0.0
    category: str = "general"


@dataclass
class EvalSummary:
    """Aggregate evaluation summary."""
    num_samples: int = 0
    avg_cer: float = 0.0
    avg_wer: float = 0.0
    latex_exact_match_rate: float = 0.0
    avg_latex_edit_distance: float = 0.0
    avg_latency: float = 0.0
    per_category: Dict = field(default_factory=dict)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    """Compute Character Error Rate (CER)."""
    if not reference:
        return 0.0 if not prediction else 1.0
    distance = levenshtein_distance(prediction, reference)
    return distance / len(reference)


def word_error_rate(prediction: str, reference: str) -> float:
    """Compute Word Error Rate (WER)."""
    pred_words = prediction.split()
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    distance = levenshtein_distance(
        " ".join(pred_words), " ".join(ref_words)
    )
    return distance / len(" ".join(ref_words))


def normalize_latex(latex: str) -> str:
    """Normalize LaTeX for comparison (ignore whitespace differences)."""
    import re
    latex = latex.strip()
    # Remove delimiters
    for d in ["$$", "$", "\\[", "\\]", "\\(", "\\)"]:
        latex = latex.replace(d, "")
    # Normalize whitespace
    latex = re.sub(r'\s+', ' ', latex).strip()
    return latex


def latex_exact_match(prediction: str, reference: str) -> bool:
    """Check if two LaTeX expressions are equivalent (normalized)."""
    return normalize_latex(prediction) == normalize_latex(reference)


def latex_edit_distance(prediction: str, reference: str) -> float:
    """Compute normalized edit distance between LaTeX expressions."""
    pred = normalize_latex(prediction)
    ref = normalize_latex(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein_distance(pred, ref) / max(len(ref), 1)


class PipelineEvaluator:
    """
    Evaluates OCR pipeline performance against ground truth.
    """

    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: OCRPipeline instance to evaluate
        """
        self.pipeline = pipeline

    def load_test_set(self, manifest_path: str) -> List[EvalSample]:
        """
        Load test set from JSONL manifest.

        Expected format per line:
        {"image_path": "...", "text": "...", "latex": "...", "category": "..."}
        """
        samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append(EvalSample(
                    image_path=data["image_path"],
                    ground_truth_text=data.get("text", ""),
                    ground_truth_latex=data.get("latex", ""),
                    category=data.get("category", "general"),
                ))
        return samples

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample."""
        start = time.time()
        result = self.pipeline.process(sample.image_path)
        latency = time.time() - start

        predicted = result.markdown.strip()

        cer = character_error_rate(predicted, sample.ground_truth_text)
        wer = word_error_rate(predicted, sample.ground_truth_text)

        em = False
        led = 0.0
        if sample.ground_truth_latex:
            em = latex_exact_match(predicted, sample.ground_truth_latex)
            led = latex_edit_distance(predicted, sample.ground_truth_latex)

        return EvalResult(
            image_path=sample.image_path,
            predicted_text=predicted,
            ground_truth_text=sample.ground_truth_text,
            cer=cer,
            wer=wer,
            latex_exact_match=em,
            latex_edit_distance=led,
            latency=latency,
            category=sample.category,
        )

    def evaluate(self, samples: List[EvalSample]) -> Tuple[EvalSummary, List[EvalResult]]:
        """
        Evaluate pipeline on a test set.

        Args:
            samples: List of EvalSample

        Returns:
            Tuple of (EvalSummary, List[EvalResult])
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
                print(f"  [{i+1}/{len(samples)}] CER={result.cer:.3f} "
                      f"WER={result.wer:.3f} Latency={result.latency:.2f}s")
            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] ERROR: {e}")

        # Compute summary
        summary = EvalSummary(
            num_samples=len(results),
            avg_cer=np.mean([r.cer for r in results]) if results else 0,
            avg_wer=np.mean([r.wer for r in results]) if results else 0,
            latex_exact_match_rate=(
                np.mean([r.latex_exact_match for r in results]) if results else 0
            ),
            avg_latex_edit_distance=(
                np.mean([r.latex_edit_distance for r in results]) if results else 0
            ),
            avg_latency=np.mean([r.latency for r in results]) if results else 0,
        )

        # Per-category breakdown
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            summary.per_category[cat] = {
                "count": len(cat_results),
                "avg_cer": float(np.mean([r.cer for r in cat_results])),
                "avg_wer": float(np.mean([r.wer for r in cat_results])),
                "latex_em_rate": float(np.mean([r.latex_exact_match for r in cat_results])),
                "avg_latency": float(np.mean([r.latency for r in cat_results])),
            }

        return summary, results


def print_summary(summary: EvalSummary, name: str = "Pipeline"):
    """Pretty-print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {name}")
    print(f"{'='*60}")
    print(f"  Samples:            {summary.num_samples}")
    print(f"  Avg CER:            {summary.avg_cer:.4f} ({summary.avg_cer*100:.2f}%)")
    print(f"  Avg WER:            {summary.avg_wer:.4f} ({summary.avg_wer*100:.2f}%)")
    print(f"  LaTeX Exact Match:  {summary.latex_exact_match_rate:.4f} ({summary.latex_exact_match_rate*100:.2f}%)")
    print(f"  Avg LaTeX EditDist: {summary.avg_latex_edit_distance:.4f}")
    print(f"  Avg Latency:        {summary.avg_latency:.3f}s")

    if summary.per_category:
        print(f"\n  Per-Category Breakdown:")
        for cat, metrics in summary.per_category.items():
            print(f"    {cat}: CER={metrics['avg_cer']:.4f} "
                  f"WER={metrics['avg_wer']:.4f} "
                  f"EM={metrics['latex_em_rate']:.2f} "
                  f"n={metrics['count']}")
    print(f"{'='*60}\n")


def main():
    """CLI for pipeline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate OCR Pipeline")
    parser.add_argument("--manifest", "-m", required=True,
                        help="Path to test manifest JSONL file")
    parser.add_argument("--output", "-o", default="evaluation/results.json",
                        help="Path to output results JSON")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no-surya", action="store_true",
                        help="Disable Surya layout detection")
    parser.add_argument("--no-unimernet", action="store_true",
                        help="Disable UniMERNet/cascading math")
    parser.add_argument("--no-latex-validator", action="store_true",
                        help="Disable LaTeX grammar validation")
    args = parser.parse_args()

    from inference.pipeline import OCRPipeline, PipelineConfig

    config = PipelineConfig(
        device=args.device,
        use_surya_layout=not args.no_surya,
        use_cascading_math=not args.no_unimernet,
        enable_latex_validation=not args.no_latex_validator,
    )
    pipeline = OCRPipeline(config)
    evaluator = PipelineEvaluator(pipeline)

    print(f"Loading test set from: {args.manifest}")
    samples = evaluator.load_test_set(args.manifest)
    print(f"Loaded {len(samples)} samples")

    print("\nRunning evaluation...")
    summary, results = evaluator.evaluate(samples)

    print_summary(summary)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
