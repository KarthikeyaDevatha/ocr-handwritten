#!/usr/bin/env python
"""
Setup and validation script for the OCR system.
Checks dependencies, downloads models, and runs basic tests.
"""

import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if all dependencies are installed."""
    print("\nChecking dependencies...")
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("onnxruntime", "ONNX Runtime"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("streamlit", "Streamlit"),
        ("fastwer", "FastWER"),
    ]
    
    missing = []
    
    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("ℹ️ CUDA not available, will use CPU")
            return True
    except Exception as e:
        print(f"ℹ️ Could not check CUDA: {e}")
        return True


def check_project_structure():
    """Verify project structure exists."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "data/raw",
        "data/synthetic",
        "data/manifests",
        "models/yolo",
        "models/trocr_text/tokenizer",
        "models/trocr_math/tokenizer",
        "training",
        "inference",
        "evaluation",
        "app",
        "outputs",
        "samples",
        "checkpoints",
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created {dir_path}")
    
    return True


def generate_sample_data():
    """Generate sample data if not exists."""
    print("\nChecking sample data...")
    
    manifest_path = PROJECT_ROOT / "data/manifests/train.jsonl"
    sample_path = PROJECT_ROOT / "samples/page.png"
    
    if not manifest_path.exists() or not sample_path.exists():
        print("Generating sample data...")
        
        script_path = PROJECT_ROOT / "data/synthesis_script.py"
        
        # Generate training data
        subprocess.run([
            sys.executable, str(script_path),
            "--num-samples", "100",
            "--output-dir", str(PROJECT_ROOT / "data/synthetic"),
            "--manifest-dir", str(PROJECT_ROOT / "data/manifests")
        ])
        
        # Generate sample page
        subprocess.run([
            sys.executable, str(script_path),
            "--generate-page"
        ])
        
        print("✅ Sample data generated")
    else:
        print("✅ Sample data exists")
    
    return True


def test_pipeline_import():
    """Test that pipeline can be imported."""
    print("\nTesting pipeline import...")
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from inference.pipeline import OCRPipeline, PipelineConfig
        print("✅ Pipeline imports successfully")
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False


def test_inference():
    """Run a basic inference test."""
    print("\nTesting inference (fallback mode)...")
    
    sample_path = PROJECT_ROOT / "samples/page.png"
    
    if not sample_path.exists():
        print("⚠️ Sample image not found, skipping inference test")
        return True
    
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from inference.pipeline import OCRPipeline, PipelineConfig
        
        config = PipelineConfig(device="cpu")
        pipeline = OCRPipeline(config)
        
        result = pipeline.process(str(sample_path))
        
        print(f"✅ Inference completed in {result.total_time:.2f}s")
        print(f"   Detected: {result.num_text_regions} text, {result.num_math_regions} math regions")
        
        if result.markdown:
            print(f"   Output length: {len(result.markdown)} characters")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Inference test failed (models may not be trained yet): {e}")
        return True  # Don't fail setup for this


def print_next_steps():
    """Print next steps for user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("""
Next steps:

1. GENERATE TRAINING DATA:
   python data/synthesis_script.py --num-samples 1000

2. TRAIN TEXT OCR:
   python training/train_trocr_text.py \\
     --manifest data/manifests/train.jsonl \\
     --epochs 10 --batch_size 8

3. TRAIN MATH OCR:
   python training/train_mfr_math.py \\
     --manifest data/manifests/train.jsonl \\
     --epochs 8 --batch_size 4

4. TRAIN LAYOUT DETECTOR:
   python training/train_yolo.py --action generate
   python training/train_yolo.py --action train --epochs 50

5. EXPORT TO ONNX:
   python training/export_onnx.py \\
     --model checkpoints/trocr_text/final \\
     --output models/trocr_text

6. RUN INFERENCE:
   python inference/pipeline.py --image samples/page.png

7. START DEMO:
   streamlit run app/app.py
""")


def main():
    """Run all setup checks."""
    print("="*60)
    print("OCR SYSTEM SETUP VALIDATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA", check_cuda),
        ("Project Structure", check_project_structure),
        ("Sample Data", generate_sample_data),
        ("Pipeline Import", test_pipeline_import),
        ("Inference Test", test_inference),
    ]
    
    all_passed = True
    
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_passed = False
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            all_passed = False
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
