"""
Math Formula Recognition (MFR) Training Script.
Fine-tunes Pix2Text-MFR model for handwritten mathematical expressions.
"""

import os
import sys
import json
import argparse
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MathDataset(Dataset):
    """Dataset for math formula recognition."""
    
    def __init__(
        self,
        manifest_path: str,
        processor: TrOCRProcessor,
        max_target_length: int = 256,
        image_size: tuple = (224, 224)
    ):
        self.processor = processor
        self.max_target_length = max_target_length
        self.image_size = image_size
        self.samples = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Only include math samples
                if sample.get('mode') in ['math', 'mixed']:
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} math samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and resize image to 224x224
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Extract LaTeX from ground truth
        text = sample['ground_truth_text']
        # Remove $ delimiters if present
        text = text.strip()
        if text.startswith('$') and text.endswith('$'):
            text = text[1:-1]
        
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Replace padding with -100 for loss masking
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def load_mfr_model(model_name: str = "microsoft/trocr-small-handwritten"):
    """
    Load or initialize MFR model.
    
    Note: breezedeus/pix2text-mfr may not be directly available.
    Fallback to TrOCR-small with math-adapted tokenizer.
    """
    try:
        # Try loading Pix2Text MFR
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        print("Using TrOCR-small as base for math recognition...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    
    return processor, model


def main():
    parser = argparse.ArgumentParser(description="Train MFR for math recognition")
    parser.add_argument("--manifest", type=str, required=True, help="Training manifest path")
    parser.add_argument("--val-manifest", type=str, help="Validation manifest path")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--output", type=str, default="checkpoints/trocr_math")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-handwritten",
                        help="Base model (default: trocr-small, use breezedeus/pix2text-mfr if available)")
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--adapt-latex", action="store_true", help="Add LaTeX tokens to tokenizer")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    processor, model = load_mfr_model(args.model_name)
    
    # Optionally adapt tokenizer for LaTeX
    if args.adapt_latex:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tokenizer_utils import STANDARD_LATEX_TOKENS
        
        num_added = processor.tokenizer.add_tokens(STANDARD_LATEX_TOKENS)
        print(f"Added {num_added} LaTeX tokens")
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        # Initialize new embeddings
        if num_added > 0:
            with torch.no_grad():
                embedding_layer = model.decoder.get_input_embeddings()
                new_embeddings = torch.randn(num_added, embedding_layer.weight.shape[1]) * 0.02
                embedding_layer.weight[-num_added:] = new_embeddings
    
    model = model.to(device)
    
    # Set decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Create datasets
    train_dataset = MathDataset(
        args.manifest,
        processor,
        max_target_length=args.max_target_length,
        image_size=(args.image_size, args.image_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_manifest:
        val_dataset = MathDataset(
            args.val_manifest,
            processor,
            max_target_length=args.max_target_length,
            image_size=(args.image_size, args.image_size)
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output, "best")
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output, f"epoch_{epoch}")
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
    
    # Save final
    final_path = os.path.join(args.output, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    # Save tokenizer separately for ONNX export
    tokenizer_path = "models/trocr_math/tokenizer"
    os.makedirs(tokenizer_path, exist_ok=True)
    processor.save_pretrained(tokenizer_path)
    
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
