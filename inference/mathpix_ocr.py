import os
import json
import base64
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class MathpixResult:
    text: str = ""
    confidence: float = 0.0
    confidence_rate: Optional[float] = None
    latex_styled: Optional[str] = None
    html: Optional[str] = None
    line_data: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    is_mock: bool = False

class MathpixOCR:
    def __init__(self):
        self.app_id = os.environ.get("MATHPIX_APP_ID", "")
        self.app_key = os.environ.get("MATHPIX_APP_KEY", "")
        self.is_mock = os.environ.get("MATHPIX_MOCK", "false").lower() == "true"
        
        # Available if we have credentials or are in mock mode
        self.is_available = bool((self.app_id and self.app_key) or self.is_mock)

    def recognize_image(self, image_path: str) -> MathpixResult:
        if self.is_mock:
            return MathpixResult(
                text="E = mc^2",
                confidence=0.95,
                confidence_rate=0.95,
                latex_styled="$$ E = mc^2 $$",
                html="<p>E = mc^2</p>",
                line_data=[],
                is_mock=True
            )
            
        if not self.is_available:
            return MathpixResult(error="Mathpix credentials not configured")

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key
            }
            
            data = {
                "options_json": json.dumps({
                    "math_inline_delimiters": ["$", "$"],
                    "rm_spaces": True
                })
            }
            
            files = {"file": image_data}
            
            response = requests.post(
                "https://api.mathpix.com/v3/text",
                headers=headers,
                data=data,
                files=files,
                timeout=15
            )
            
            result_json = response.json() if response.content else {}
            
            if response.status_code == 200:
                if "error" in result_json:
                    return MathpixResult(error=result_json["error"])
                    
                text = result_json.get("text", "")
                confidence = result_json.get("confidence", 0.0)
                confidence_rate = result_json.get("confidence_rate", confidence)
                latex_styled = result_json.get("latex_styled", "")
                html = result_json.get("html", "")
                line_data = result_json.get("line_data", [])
                
                return MathpixResult(
                    text=text,
                    confidence=confidence,
                    confidence_rate=confidence_rate,
                    latex_styled=latex_styled,
                    html=html,
                    line_data=line_data
                )
            else:
                error_msg = result_json.get("error", "Unknown error")
                if isinstance(error_msg, dict):
                    error_msg = json.dumps(error_msg)
                return MathpixResult(error=f"HTTP {response.status_code}: {error_msg}")
                
        except Exception as e:
            return MathpixResult(error=str(e))
