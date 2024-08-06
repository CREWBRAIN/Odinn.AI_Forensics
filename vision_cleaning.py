# 🧊 Odinn AI Forensics Tool / vision_cleaning.py

# vision_cleaning.py

import base64
import logging

import litellm
from litellm import completion

from rich.console import Console

from settings import Settings

console = Console()

class VisionCleanProcessor:
    """Processes an image and a text file by sending a vision model using litellm."""

    def __init__(self, settings):
        self.settings = settings

    def retrieve_image(self, image_path: str) -> str:
        """Retrieves an image from a path and converts it to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            console.print(f"[bold red]Error loading image: {e}[/]")
            return ""

    def retrieve_text(self, text_path: str) -> str:
        """Retrieves text from a path and returns it."""
        with open(text_path, "r", encoding='utf-8') as text_file:
            return text_file.read()

    def send_to_litellm(self, image_base64: str, text: str, metadata: dict) -> str:
        """Sends an image and text to litellm and returns the response."""
        # Construct the prompt with metadata
        prompt = f"Restore this text to its original detail and turn this into utf-8 plain text: {text}\n\n"
        prompt += "Metadata:\n"
        for key, value in metadata.items():
            prompt += f"- {key}: {value}\n"

        # Set litellm API key if provided in settings
        if self.settings.vision_api_key:
            litellm.api_key = self.settings.vision_api_key

        response = litellm.completion(
            model=self.settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=self.settings.vision_temperature,
            max_tokens=self.settings.vision_max_tokens
        )
        return response.choices[0].message.content if response.choices else None

def process_vision_clean(image_path: str, text_path: str, metadata: dict, settings: Settings) -> str:
    """Processes an image and a text file by sending a vision model using litellm."""
    processor = VisionCleanProcessor(settings)
    image_base64 = processor.retrieve_image(image_path)
    text = processor.retrieve_text(text_path)
    return processor.send_to_litellm(image_base64, text, metadata)