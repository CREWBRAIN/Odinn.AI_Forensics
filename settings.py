# 🧊 Odinn AI Forensics Tool / settings.py

import json
import os
from typing import List, Dict, Any, Optional
import torch

from rich.console import Console
from rich.panel import Panel

# Import marker settings
from marker.settings import settings as marker_settings

console = Console()

class Settings:
    def __init__(self):
        # --- Marker Settings ---
        self.torch_device = self._detect_torch_device()
        self.ocr_engine = marker_settings.OCR_ENGINE  # Use marker's default
        self.langs: Optional[List[str]] = marker_settings.DEFAULT_LANG.split(",")  # Use marker's default
        self.batch_multiplier = 2
        self.extract_bboxes = False
        self.bbox_types: List[str] = []
        self.visualize_bboxes = True

        # Additional Marker Settings (add as needed)
        self.ocr_all_pages = marker_settings.OCR_ALL_PAGES
        self.paginate_output = marker_settings.PAGINATE_OUTPUT
        self.extract_images = marker_settings.EXTRACT_IMAGES
        self.bad_span_types = marker_settings.BAD_SPAN_TYPES

        # ADDED: DETECTOR_MODEL_CHECKPOINT
        self.DETECTOR_MODEL_CHECKPOINT = "vikp/surya_detector"  # Default checkpoint for the detector model

        # --- Vision Cleaning Settings ---
        self.vision_model = "ollama/llava-phi3"
        self.vision_api_key: Optional[str] = None  # Add if using a paid API
        self.vision_temperature = 0.7  # Example: Control randomness of LLaVA output
        self.vision_max_tokens = 512  # Example: Limit the length of LLaVA output

        # --- Output Settings ---
        self.output_folder = "output"
        self.citation: Dict[str, str] = {}

    def _detect_torch_device(self) -> str:
        """Auto-detects the best available torch device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Loads settings from a dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, filepath: str) -> None:
        """Saves settings to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def __str__(self) -> str:
        """Returns a string representation of the settings."""
        return f"""
Torch Device:       {self.torch_device}
OCR Engine:         {self.ocr_engine}
Languages:          {self.langs}
Batch Multiplier:   {self.batch_multiplier}
Extract BBoxes:     {self.extract_bboxes}
BBox Types:         {self.bbox_types}
Visualize BBoxes:   {self.visualize_bboxes}
OCR All Pages:      {self.ocr_all_pages}
Paginate Output:    {self.paginate_output}
Extract Images:     {self.extract_images}
Bad Span Types:     {self.bad_span_types}
Vision Model:       {self.vision_model}
Vision API Key:     {self.vision_api_key}
Vision Temperature: {self.vision_temperature}
Vision Max Tokens:  {self.vision_max_tokens}
Output Folder:      {self.output_folder}
Citation:           {self.citation}
"""

    def set_output_folder(self, folder: str):
        self.output_folder = folder
        os.makedirs(self.output_folder, exist_ok=True)