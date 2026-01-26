"""
TrOCR Text Fine-Tuning Script for Handwritten English Recognition.
Uses microsoft/trocr-small-handwritten with LaTeX-adapted tokenizer.
"""

import os
import sys
import json
import argparse
import random
from typing import Optional

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

# Import tokenizer utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_utils import adapt_tokenizer, STANDARD_LATEX_TOKENS


# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class OCRDataset(Dataset):
    """Dataset for OCR training from JSONL manifest."""
    
    def __init__(
        self,
        manifest_path: str,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        mode_filter: Optional[str] = None
    ):
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                if mode_filter is None or sample.get('mode') == mode_filter:
                    self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Process text
        text = sample['ground_truth_text']
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Replace padding token id with -100 for loss masking
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
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train TrOCR for text recognition")
    parser.add_argument("--manifest", type=str, required=True, help="Path to training manifest")
    parser.add_argument("--val-manifest", type=str, help="Path to validation manifest")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--output", type=str, default="checkpoints/trocr_text", help="Output directory")
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-handwritten")
    parser.add_argument("--adapt-tokenizer", action="store_true", help="Adapt tokenizer with LaTeX tokens")
    parser.add_argument("--tokenizer-dir", type=str, default="models/trocr_text/tokenizer")
    parser.add_argument("--mode-filter", type=str, default="text", help="Filter samples by mode")
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load or adapt tokenizer and model
    if args.adapt_tokenizer:
        print("Adapting tokenizer with LaTeX tokens...")
        processor, model = adapt_tokenizer(
            model_name=args.model_name,
            latex_tokens=STANDARD_LATEX_TOKENS,
            output_dir=args.tokenizer_dir
        )
    else:
        print(f"Loading model and processor from {args.model_name}...")
        processor = TrOCRProcessor.from_pretrained(args.model_name, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
        
        # Load adapted tokenizer if available and valid
        tokenizer_valid = (
            os.path.exists(args.tokenizer_dir) and
            os.path.exists(os.path.join(args.tokenizer_dir, "preprocessor_config.json"))
        )
        if tokenizer_valid:
            print(f"Loading adapted tokenizer from {args.tokenizer_dir}")
            processor = TrOCRProcessor.from_pretrained(args.tokenizer_dir, use_fast=False)
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    model = model.to(device)
    
    # Set decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Create datasets
    train_dataset = OCRDataset(
        args.manifest,
        processor,
        max_target_length=args.max_target_length,
        mode_filter=args.mode_filter if args.mode_filter != "all" else None
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
        val_dataset = OCRDataset(
            args.val_manifest,
            processor,
            max_target_length=args.max_target_length,
            mode_filter=args.mode_filter if args.mode_filter != "all" else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output, "best")
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                print(f"Saved best model to {best_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output, f"epoch_{epoch}")
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
