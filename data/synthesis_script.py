"""
Synthetic data generation for mixed handwritten text and math expressions.
Generates training samples with ground truth annotations.
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import string


# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Sample text fragments
TEXT_FRAGMENTS = [
    "The derivative of",
    "Consider the function",
    "We can solve this by",
    "Let x be a variable",
    "Given that",
    "From the equation",
    "Substituting into",
    "Therefore we have",
    "This implies that",
    "By integration",
    "Taking the limit as",
    "Using the formula",
    "We obtain",
    "Note that",
    "It follows that",
    "Suppose that",
    "Assume",
    "Proof:",
    "Example:",
    "Solution:",
]

# Sample LaTeX expressions
MATH_EXPRESSIONS = [
    r"$x^2 + y^2 = r^2$",
    r"$\int_0^1 x^2 dx$",
    r"$\frac{dy}{dx} = 2x$",
    r"$\sum_{n=1}^{\infty} \frac{1}{n^2}$",
    r"$\sqrt{a^2 + b^2}$",
    r"$e^{i\pi} + 1 = 0$",
    r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$",
    r"$\alpha + \beta = \gamma$",
    r"$\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}$",
    r"$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$",
    r"$\frac{\partial f}{\partial x}$",
    r"$\binom{n}{k} = \frac{n!}{k!(n-k)!}$",
    r"$\vec{F} = m\vec{a}$",
    r"$E = mc^2$",
    r"$\lambda = \frac{h}{p}$",
    r"$\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{N}}$",
    r"$A = \pi r^2$",
    r"$V = \frac{4}{3}\pi r^3$",
    r"$\cos^2\theta + \sin^2\theta = 1$",
    r"$\log_a(xy) = \log_a x + \log_a y$",
]

# Mixed line templates
MIXED_TEMPLATES = [
    ("Calculate the integral ", r"$\int x^2 dx$"),
    ("The solution is ", r"$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$"),
    ("We know that ", r"$a^2 + b^2 = c^2$"),
    ("From ", r"$F = ma$", " we get"),
    ("Given ", r"$y = mx + b$", " find slope"),
    ("Evaluate ", r"$\lim_{x \to 0} \frac{\sin x}{x}$"),
    ("The derivative is ", r"$f'(x) = 2x$"),
]


class SyntheticDataGenerator:
    """Generates synthetic handwritten-style training data."""
    
    def __init__(
        self,
        output_dir: str = "data/synthetic",
        image_size: Tuple[int, int] = (800, 100),
        font_size: int = 32
    ):
        self.output_dir = output_dir
        self.image_size = image_size
        self.font_size = font_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to load a handwriting-like font, fallback to default
        self.font = self._load_font()
    
    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load a font for text rendering."""
        font_paths = [
            "/System/Library/Fonts/Marker Felt.ttc",
            "/System/Library/Fonts/Noteworthy.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, self.font_size)
                except Exception:
                    continue
        
        return ImageFont.load_default()
    
    def _add_noise(self, image: Image.Image, intensity: float = 0.1) -> Image.Image:
        """Add slight noise to simulate handwriting imperfections."""
        img_array = np.array(image)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def _add_distortion(self, image: Image.Image) -> Image.Image:
        """Add slight geometric distortion."""
        # Simple random rotation
        angle = random.uniform(-2, 2)
        return image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    
    def generate_text_line(self, text: str) -> Tuple[Image.Image, str]:
        """Generate an image of a text line."""
        # Create image with white background
        img = Image.new('RGB', self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Random starting position
        x = random.randint(10, 30)
        y = (self.image_size[1] - self.font_size) // 2 + random.randint(-5, 5)
        
        # Draw text in dark color with slight variation
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        draw.text((x, y), text, font=self.font, fill=color)
        
        # Add noise and distortion
        img = self._add_noise(img, random.uniform(0.02, 0.08))
        img = self._add_distortion(img)
        
        return img, text
    
    def generate_math_line(self, latex: str) -> Tuple[Image.Image, str]:
        """Generate an image with LaTeX expression rendered as text."""
        # For synthetic data, we render the LaTeX source as text
        # In real training, this would be actual rendered math
        img = Image.new('RGB', self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        x = random.randint(10, 30)
        y = (self.image_size[1] - self.font_size) // 2 + random.randint(-5, 5)
        
        # Display LaTeX without $ delimiters for rendering
        display_text = latex.strip('$')
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        draw.text((x, y), display_text, font=self.font, fill=color)
        
        img = self._add_noise(img, random.uniform(0.02, 0.08))
        img = self._add_distortion(img)
        
        return img, latex
    
    def generate_mixed_line(self, template: Tuple) -> Tuple[Image.Image, str]:
        """Generate an image with mixed text and math."""
        full_text = ''.join(template)
        img = Image.new('RGB', self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        x = random.randint(10, 30)
        y = (self.image_size[1] - self.font_size) // 2 + random.randint(-5, 5)
        
        # Render the mixed content
        display_text = full_text.replace('$', '')
        color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        draw.text((x, y), display_text, font=self.font, fill=color)
        
        img = self._add_noise(img, random.uniform(0.02, 0.08))
        img = self._add_distortion(img)
        
        return img, full_text
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        text_ratio: float = 0.4,
        math_ratio: float = 0.4,
        mixed_ratio: float = 0.2
    ) -> List[Dict]:
        """Generate a complete dataset with manifests."""
        
        assert abs(text_ratio + math_ratio + mixed_ratio - 1.0) < 0.01, "Ratios must sum to 1"
        
        num_text = int(num_samples * text_ratio)
        num_math = int(num_samples * math_ratio)
        num_mixed = num_samples - num_text - num_math
        
        samples = []
        
        print(f"Generating {num_text} text samples...")
        for i in range(num_text):
            text = random.choice(TEXT_FRAGMENTS)
            img, gt = self.generate_text_line(text)
            
            filename = f"text_{i:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            
            samples.append({
                "image_path": filepath,
                "ground_truth_text": gt,
                "mode": "text"
            })
        
        print(f"Generating {num_math} math samples...")
        for i in range(num_math):
            latex = random.choice(MATH_EXPRESSIONS)
            img, gt = self.generate_math_line(latex)
            
            filename = f"math_{i:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            
            samples.append({
                "image_path": filepath,
                "ground_truth_text": gt,
                "mode": "math"
            })
        
        print(f"Generating {num_mixed} mixed samples...")
        for i in range(num_mixed):
            template = random.choice(MIXED_TEMPLATES)
            img, gt = self.generate_mixed_line(template)
            
            filename = f"mixed_{i:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            
            samples.append({
                "image_path": filepath,
                "ground_truth_text": gt,
                "mode": "mixed"
            })
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    def save_manifests(
        self,
        samples: List[Dict],
        manifest_dir: str = "data/manifests",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ):
        """Split samples and save to train/val/test manifests."""
        os.makedirs(manifest_dir, exist_ok=True)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        for name, data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            filepath = os.path.join(manifest_dir, f"{name}.jsonl")
            with open(filepath, 'w') as f:
                for sample in data:
                    f.write(json.dumps(sample) + '\n')
            print(f"Saved {len(data)} samples to {filepath}")


def generate_sample_page(output_path: str = "samples/page.png"):
    """Generate a sample page image for testing the pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a page-sized image
    page_size = (800, 1200)
    img = Image.new('RGB', page_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    font_size = 28
    font_paths = [
        "/System/Library/Fonts/Marker Felt.ttc",
        "/System/Library/Fonts/Noteworthy.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = ImageFont.load_default()
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
    
    # Sample content for the page
    lines = [
        "Math Notes - Chapter 5",
        "",
        "The quadratic formula is:",
        "$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$",
        "",
        "Example: Solve x^2 + 5x + 6 = 0",
        "",
        "Using the formula with a=1, b=5, c=6:",
        "$x = \\frac{-5 \\pm \\sqrt{25-24}}{2}$",
        "$x = \\frac{-5 \\pm 1}{2}$",
        "",
        "Therefore x = -2 or x = -3",
        "",
        "Integration example:",
        "$\\int x^2 dx = \\frac{x^3}{3} + C$",
        "",
        "Note: Remember to add constant C",
    ]
    
    y = 50
    for line in lines:
        x = 50 + random.randint(-5, 5)
        color = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))
        draw.text((x, y), line.replace('$', ''), font=font, fill=color)
        y += 60
    
    # Add some noise
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy)
    
    img.save(output_path)
    print(f"Generated sample page: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests")
    parser.add_argument("--generate-page", action="store_true", help="Generate sample page")
    
    args = parser.parse_args()
    
    if args.generate_page:
        generate_sample_page()
    else:
        generator = SyntheticDataGenerator(output_dir=args.output_dir)
        samples = generator.generate_dataset(num_samples=args.num_samples)
        generator.save_manifests(samples, manifest_dir=args.manifest_dir)
        print(f"Generated {len(samples)} samples")
