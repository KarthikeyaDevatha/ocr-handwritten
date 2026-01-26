"""
Tokenizer adaptation utilities for LaTeX token integration.
Adds atomic LaTeX tokens to TrOCR tokenizer for better math recognition.
"""

import os
import json
import re
from collections import Counter
from typing import List, Optional, Tuple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


# Standard LaTeX tokens that should be atomic
STANDARD_LATEX_TOKENS = [
    # Greek letters
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
    "\\zeta", "\\eta", "\\theta", "\\vartheta", "\\iota", "\\kappa",
    "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\varpi", "\\rho",
    "\\varrho", "\\sigma", "\\varsigma", "\\tau", "\\upsilon", "\\phi",
    "\\varphi", "\\chi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi",
    "\\Sigma", "\\Upsilon", "\\Phi", "\\Psi", "\\Omega",
    # Operators and relations
    "\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", "\\oint",
    "\\partial", "\\nabla", "\\infty", "\\pm", "\\mp", "\\times",
    "\\div", "\\cdot", "\\ast", "\\star", "\\circ", "\\bullet",
    "\\oplus", "\\otimes", "\\odot", "\\oslash", "\\ominus",
    "\\le", "\\leq", "\\ge", "\\geq", "\\neq", "\\approx", "\\equiv",
    "\\sim", "\\simeq", "\\cong", "\\propto", "\\ll", "\\gg",
    "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\in", "\\ni",
    "\\notin", "\\cap", "\\cup", "\\setminus", "\\emptyset",
    "\\forall", "\\exists", "\\nexists", "\\neg", "\\land", "\\lor",
    "\\implies", "\\iff", "\\therefore", "\\because",
    # Functions
    "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc",
    "\\arcsin", "\\arccos", "\\arctan", "\\sinh", "\\cosh", "\\tanh",
    "\\log", "\\ln", "\\exp", "\\lim", "\\limsup", "\\liminf",
    "\\min", "\\max", "\\sup", "\\inf", "\\arg", "\\det", "\\dim",
    "\\gcd", "\\lcm", "\\mod", "\\bmod",
    # Delimiters
    "\\left", "\\right", "\\langle", "\\rangle", "\\lfloor", "\\rfloor",
    "\\lceil", "\\rceil", "\\lvert", "\\rvert", "\\lVert", "\\rVert",
    # Accents and decorations
    "\\hat", "\\bar", "\\vec", "\\dot", "\\ddot", "\\tilde", "\\widehat",
    "\\overline", "\\underline", "\\overbrace", "\\underbrace",
    # Spacing and formatting
    "\\quad", "\\qquad", "\\text", "\\mathrm", "\\mathbf", "\\mathit",
    "\\mathsf", "\\mathcal", "\\mathbb", "\\mathfrak",
    # Matrices and arrays
    "\\begin", "\\end", "\\matrix", "\\pmatrix", "\\bmatrix", "\\vmatrix",
    "\\cases", "\\array", "\\align", "\\aligned",
    # Special symbols
    "\\ldots", "\\cdots", "\\vdots", "\\ddots", "\\prime", "\\dagger",
    "\\ddagger", "\\ell", "\\hbar", "\\imath", "\\jmath",
    # Arrows
    "\\to", "\\gets", "\\leftarrow", "\\rightarrow", "\\leftrightarrow",
    "\\Leftarrow", "\\Rightarrow", "\\Leftrightarrow",
    "\\uparrow", "\\downarrow", "\\updownarrow",
    "\\mapsto", "\\longmapsto", "\\hookrightarrow", "\\hookleftarrow",
]


def extract_latex_commands(corpus_path: str, min_freq: int = 500) -> List[str]:
    """
    Scan a LaTeX corpus and extract frequently occurring commands.
    
    Args:
        corpus_path: Path to file or directory containing LaTeX expressions
        min_freq: Minimum frequency threshold for inclusion
    
    Returns:
        List of LaTeX command strings
    """
    latex_pattern = re.compile(r'\\[a-zA-Z]+')
    command_counts = Counter()
    
    if os.path.isfile(corpus_path):
        files = [corpus_path]
    else:
        files = []
        for root, _, filenames in os.walk(corpus_path):
            for f in filenames:
                if f.endswith(('.txt', '.tex', '.json', '.jsonl')):
                    files.append(os.path.join(root, f))
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Handle JSONL format
                if filepath.endswith('.jsonl'):
                    for line in content.strip().split('\n'):
                        try:
                            data = json.loads(line)
                            text = data.get('ground_truth_text', '') or data.get('latex', '') or data.get('formula', '')
                            commands = latex_pattern.findall(text)
                            command_counts.update(commands)
                        except json.JSONDecodeError:
                            continue
                else:
                    commands = latex_pattern.findall(content)
                    command_counts.update(commands)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
    
    # Filter by frequency
    frequent_commands = [cmd for cmd, count in command_counts.items() if count >= min_freq]
    
    # Combine with standard tokens, removing duplicates
    all_tokens = list(set(STANDARD_LATEX_TOKENS + frequent_commands))
    
    return sorted(all_tokens)


def adapt_tokenizer(
    model_name: str = "microsoft/trocr-small-handwritten",
    latex_tokens: Optional[List[str]] = None,
    output_dir: str = "models/trocr_text/tokenizer",
    init_scale: float = 0.02
) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """
    Adapt TrOCR tokenizer by adding LaTeX tokens and resize model embeddings.
    
    Args:
        model_name: HuggingFace model name
        latex_tokens: List of LaTeX tokens to add (uses defaults if None)
        output_dir: Directory to save adapted tokenizer
        init_scale: Scale for random normal initialization of new embeddings
    
    Returns:
        Tuple of (adapted processor, adapted model)
    """
    if latex_tokens is None:
        latex_tokens = STANDARD_LATEX_TOKENS
    
    print(f"Loading processor from {model_name}...")
    processor = TrOCRProcessor.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Get original vocabulary size
    original_vocab_size = len(processor.tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Add new tokens
    num_added = processor.tokenizer.add_tokens(latex_tokens)
    print(f"Added {num_added} new LaTeX tokens")
    
    # Resize model embeddings
    new_vocab_size = len(processor.tokenizer)
    model.decoder.resize_token_embeddings(new_vocab_size)
    print(f"New vocabulary size: {new_vocab_size}")
    
    # Initialize new embeddings with scaled random normal
    if num_added > 0:
        with torch.no_grad():
            # Get embedding layer
            embedding_layer = model.decoder.get_input_embeddings()
            
            # Initialize new token embeddings
            new_embeddings = torch.randn(num_added, embedding_layer.weight.shape[1]) * init_scale
            embedding_layer.weight[-num_added:] = new_embeddings
            
            # Also update output projection if it exists
            if hasattr(model.decoder, 'lm_head'):
                lm_head = model.decoder.lm_head
                if hasattr(lm_head, 'weight'):
                    new_lm_weights = torch.randn(num_added, lm_head.weight.shape[1]) * init_scale
                    lm_head.weight.data[-num_added:] = new_lm_weights
    
    # Save adapted tokenizer
    os.makedirs(output_dir, exist_ok=True)
    processor.tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved adapted tokenizer to {output_dir}")
    
    return processor, model


def load_adapted_processor(tokenizer_dir: str) -> TrOCRProcessor:
    """Load an adapted processor from disk."""
    return TrOCRProcessor.from_pretrained(tokenizer_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adapt tokenizer with LaTeX tokens")
    parser.add_argument("--corpus", type=str, help="Path to LaTeX corpus for token extraction")
    parser.add_argument("--min-freq", type=int, default=500, help="Minimum frequency for extracted tokens")
    parser.add_argument("--model", type=str, default="microsoft/trocr-small-handwritten")
    parser.add_argument("--output", type=str, default="models/trocr_text/tokenizer")
    
    args = parser.parse_args()
    
    if args.corpus:
        print(f"Extracting LaTeX tokens from {args.corpus}...")
        tokens = extract_latex_commands(args.corpus, args.min_freq)
        print(f"Found {len(tokens)} LaTeX tokens")
    else:
        tokens = STANDARD_LATEX_TOKENS
        print(f"Using {len(tokens)} standard LaTeX tokens")
    
    processor, model = adapt_tokenizer(
        model_name=args.model,
        latex_tokens=tokens,
        output_dir=args.output
    )
    
    print("Tokenizer adaptation complete!")
