"""
Document reconstruction module.
Combines OCR results into final Markdown + LaTeX output.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LineResult:
    """Result for a single detected line."""
    text: str
    line_type: str  # 'text' or 'math'
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_display_math: bool = False
    
    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


class DocumentReconstructor:
    """Reconstructs document from OCR results."""
    
    def __init__(
        self,
        line_gap_threshold: float = 1.5,  # Multiplier of average line height
        paragraph_gap_threshold: float = 2.5
    ):
        """
        Initialize reconstructor.
        
        Args:
            line_gap_threshold: Gap ratio to insert newline
            paragraph_gap_threshold: Gap ratio to insert paragraph break
        """
        self.line_gap_threshold = line_gap_threshold
        self.paragraph_gap_threshold = paragraph_gap_threshold
    
    def _compute_line_heights(self, results: List[LineResult]) -> float:
        """Compute average line height."""
        if not results:
            return 30  # Default
        
        heights = [r.bbox[3] - r.bbox[1] for r in results]
        return sum(heights) / len(heights)
    
    def _format_math(self, latex: str, is_display: bool) -> str:
        """
        Format LaTeX expression with appropriate delimiters.
        
        Args:
            latex: Raw LaTeX string
            is_display: Whether to use display math
        
        Returns:
            Formatted math string
        """
        latex = latex.strip()
        
        # Remove existing delimiters
        for delim in ['$$', '$', '\\[', '\\]', '\\(', '\\)']:
            latex = latex.replace(delim, '')
        
        latex = latex.strip()
        
        if is_display:
            return f"$$\n{latex}\n$$"
        else:
            return f"${latex}$"
    
    def _merge_inline_text_and_math(self, results: List[LineResult]) -> str:
        """
        Merge text and inline math on the same line.
        
        Args:
            results: List of results on the same line
        
        Returns:
            Merged text
        """
        # Sort by x position
        sorted_results = sorted(results, key=lambda r: r.bbox[0])
        
        parts = []
        for r in sorted_results:
            if r.line_type == 'math' and not r.is_display_math:
                parts.append(f"${r.text.strip('$')}$")
            else:
                parts.append(r.text)
        
        return ' '.join(parts)
    
    def reconstruct(
        self,
        results: List[LineResult],
        title: Optional[str] = None
    ) -> str:
        """
        Reconstruct document from OCR results.
        
        Args:
            results: List of LineResult objects
            title: Optional document title
        
        Returns:
            Markdown + LaTeX formatted document
        """
        if not results:
            return ""
        
        # Sort by y position
        sorted_results = sorted(results, key=lambda r: r.center_y)
        
        avg_height = self._compute_line_heights(sorted_results)
        
        lines = []
        
        # Start with title if provided
        if title:
            lines.append(f"# {title}\n")
        
        prev_y = None
        current_line_parts = []
        
        for result in sorted_results:
            current_y = result.center_y
            
            # Check if this is a new line
            if prev_y is not None:
                gap = current_y - prev_y
                
                if gap > avg_height * self.paragraph_gap_threshold:
                    # Paragraph break
                    if current_line_parts:
                        lines.append(self._merge_inline_text_and_math(current_line_parts))
                        current_line_parts = []
                    lines.append("")  # Empty line for paragraph
                elif gap > avg_height * self.line_gap_threshold:
                    # New line
                    if current_line_parts:
                        lines.append(self._merge_inline_text_and_math(current_line_parts))
                        current_line_parts = []
            
            # Handle display math separately
            if result.line_type == 'math' and result.is_display_math:
                if current_line_parts:
                    lines.append(self._merge_inline_text_and_math(current_line_parts))
                    current_line_parts = []
                lines.append("")
                lines.append(self._format_math(result.text, is_display=True))
                lines.append("")
            else:
                current_line_parts.append(result)
            
            prev_y = current_y
        
        # Flush remaining parts
        if current_line_parts:
            lines.append(self._merge_inline_text_and_math(current_line_parts))
        
        return '\n'.join(lines)
    
    def format_as_markdown(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Add markdown formatting and metadata.
        
        Args:
            text: Reconstructed text
            metadata: Optional metadata to include
        
        Returns:
            Complete markdown document
        """
        parts = []
        
        # Add metadata as front matter
        if metadata:
            parts.append("---")
            for key, value in metadata.items():
                parts.append(f"{key}: {value}")
            parts.append("---\n")
        
        parts.append(text)
        
        return '\n'.join(parts)


def balance_braces(latex: str) -> str:
    """
    Balance curly braces in LaTeX string.
    
    Args:
        latex: Input LaTeX
    
    Returns:
        Brace-balanced LaTeX
    """
    open_count = 0
    result = []
    
    for char in latex:
        if char == '{':
            open_count += 1
            result.append(char)
        elif char == '}':
            if open_count > 0:
                open_count -= 1
                result.append(char)
            # Skip unmatched closing braces
        else:
            result.append(char)
    
    # Add missing closing braces
    result.extend(['}'] * open_count)
    
    return ''.join(result)


def cleanup_latex(latex: str) -> str:
    """
    Clean up common LaTeX issues.
    
    Args:
        latex: Raw LaTeX string
    
    Returns:
        Cleaned LaTeX
    """
    latex = latex.strip()
    
    # Balance braces
    latex = balance_braces(latex)
    
    # Fix common OCR errors
    replacements = [
        (r'\\frac\s+{', r'\\frac{'),
        (r'\\sqrt\s+{', r'\\sqrt{'),
        (r'\\sum\s+_', r'\\sum_'),
        (r'\\int\s+_', r'\\int_'),
        (r'\\lim\s+_', r'\\lim_'),
        (r'\\\s+', r'\\'),  # Remove space after backslash
        (r'{\s+', r'{'),
        (r'\s+}', r'}'),
    ]
    
    for pattern, replacement in replacements:
        latex = re.sub(pattern, replacement, latex)
    
    return latex


def detect_math_regions(text: str) -> List[Tuple[int, int, bool]]:
    """
    Detect math regions in text.
    
    Args:
        text: Input text
    
    Returns:
        List of (start, end, is_display) tuples
    """
    regions = []
    
    # Display math: $$...$$
    for match in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
        regions.append((match.start(), match.end(), True))
    
    # Inline math: $...$
    for match in re.finditer(r'\$([^$]+)\$', text):
        # Check if not part of display math
        if not any(start <= match.start() < end for start, end, _ in regions):
            regions.append((match.start(), match.end(), False))
    
    return sorted(regions, key=lambda x: x[0])


def merge_results(
    text_results: List[dict],
    math_results: List[dict]
) -> List[LineResult]:
    """
    Merge text and math OCR results.
    
    Args:
        text_results: Results from text OCR
        math_results: Results from math OCR
    
    Returns:
        Combined list of LineResult objects
    """
    all_results = []
    
    for r in text_results:
        all_results.append(LineResult(
            text=r['text'],
            line_type='text',
            confidence=r.get('confidence', 0.9),
            bbox=r['bbox']
        ))
    
    for r in math_results:
        is_display = r.get('is_display_math', False)
        all_results.append(LineResult(
            text=r['latex'],
            line_type='math',
            confidence=r.get('confidence', 0.9),
            bbox=r['bbox'],
            is_display_math=is_display
        ))
    
    return all_results
