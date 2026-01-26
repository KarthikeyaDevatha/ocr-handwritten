#!/usr/bin/env python
"""
ONNX Export Script for OCR models.
Exports TrOCR models to ONNX format with INT8 quantization.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def export_with_optimum(
    model_path: str,
    output_dir: str,
    task: str = "image-to-text-with-past",
    quantize: bool = True,
    device: str = "cpu"
) -> bool:
    """
    Export model using optimum-cli.
    
    Args:
        model_path: Path to the HuggingFace model
        output_dir: Output directory for ONNX files
        task: Export task type
        quantize: Whether to apply INT8 quantization
        device: Export device
    
    Returns:
        True if successful
    """
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_path,
        "--task", task,
        output_dir
    ]
    
    if quantize:
        cmd.extend(["--optimize", "O2"])  # Optimization level
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def export_with_transformers(
    model_path: str,
    output_dir: str,
    model_type: str = "encoder-decoder"
) -> bool:
    """
    Manual ONNX export using transformers + torch.
    
    Args:
        model_path: Path to the model
        output_dir: Output directory
        model_type: Type of model architecture
    
    Returns:
        True if successful
    """
    try:
        import torch
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        
        print(f"Loading model from {model_path}...")
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        processor = TrOCRProcessor.from_pretrained(model_path)
        
        model.eval()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export encoder
        print("Exporting encoder...")
        
        # Create dummy input
        dummy_pixel_values = torch.randn(1, 3, 384, 384)
        
        encoder_path = os.path.join(output_dir, "encoder_model.onnx")
        
        # Get encoder
        encoder = model.encoder
        
        torch.onnx.export(
            encoder,
            (dummy_pixel_values,),
            encoder_path,
            input_names=["pixel_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        print(f"Encoder exported to {encoder_path}")
        
        # Export decoder (simplified - without past key values for now)
        print("Exporting decoder...")
        
        decoder_path = os.path.join(output_dir, "decoder_model.onnx")
        
        # Decoder wrapper for clean export
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
            
            def forward(self, input_ids, attention_mask, encoder_hidden_states):
                outputs = self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states
                )
                return outputs.logits
        
        decoder_wrapper = DecoderWrapper(model.decoder)
        
        dummy_input_ids = torch.tensor([[processor.tokenizer.cls_token_id]])
        dummy_metrics_mask = torch.ones_like(dummy_input_ids)
        dummy_encoder_output = torch.randn(1, 577, 384)  # ViT output shape
        
        torch.onnx.export(
            decoder_wrapper,
            (dummy_input_ids, dummy_metrics_mask, dummy_encoder_output),
            decoder_path,
            input_names=["input_ids", "attention_mask", "encoder_hidden_states"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        print(f"Decoder exported to {decoder_path}")
        
        # Save tokenizer
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        processor.save_pretrained(tokenizer_dir)
        # Also save in root for compatibility
        processor.save_pretrained(output_dir)
        
        print(f"Tokenizer saved to {tokenizer_dir}")
        
        return True
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quantize_onnx(
    model_path: str,
    output_path: str = None,
    quantization_type: str = "dynamic"
) -> bool:
    """
    Apply INT8 quantization to ONNX model.
    
    Args:
        model_path: Path to ONNX model
        output_path: Output path (defaults to replacing input)
        quantization_type: 'dynamic' or 'static'
    
    Returns:
        True if successful
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if output_path is None:
            output_path = model_path.replace('.onnx', '_int8.onnx')
        
        print(f"Quantizing {model_path}...")
        
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        
        # Check file sizes
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"Original: {orig_size:.2f} MB")
        print(f"Quantized: {quant_size:.2f} MB")
        print(f"Reduction: {(1 - quant_size/orig_size) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export OCR models to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace name")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--method", choices=["optimum", "manual"], default="manual")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--task", type=str, default="image-to-text-with-past")
    
    args = parser.parse_args()
    
    success = False
    
    if args.method == "optimum":
        success = export_with_optimum(
            args.model,
            args.output,
            task=args.task,
            quantize=args.quantize
        )
    else:
        success = export_with_transformers(
            args.model,
            args.output
        )
        
        if success and args.quantize:
            for model_file in ["encoder_model.onnx", "decoder_model.onnx"]:
                model_path = os.path.join(args.output, model_file)
                if os.path.exists(model_path):
                    quantize_onnx(model_path)
    
    if success:
        print("\n✅ Export completed successfully!")
        print(f"Output directory: {args.output}")
    else:
        print("\n❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
