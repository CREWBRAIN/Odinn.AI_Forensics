# AI Forensics Single PDF to Markdown / Downloads models then caches them on first run

import os
import base64
import json
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
import io

import pypdfium2 as pdfium
from PIL import Image
from pdf2image import convert_from_path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, track

from imports.convert import convert_single_pdf_md_only
from imports.logger import configure_logging
from imports.models import load_all_models

import litellm
from litellm import completion

configure_logging(log_level=logging.DEBUG)  # Set log level to DEBUG for verbose logging
console = Console()

def log_decorator(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Completed {func.__name__}")
        return result
    return wrapper

def get_subfolder_path(out_folder: str, fname: str) -> str:
    base_name = os.path.basename(fname).rsplit(".", 1)[0]
    return os.path.join(out_folder, base_name)

def save_file(content: str, path: str, mode: str = "w", encoding: str = "utf-8") -> None:
    with open(path, mode, encoding=encoding) as f:
        f.write(content)

def save_image(image: Image.Image, path: str) -> None:
    try:
        image.save(path, "JPEG")
        logging.debug(f"Saved image to {path}")
    except Exception as e:
        logging.error(f"Error saving image to {path}: {e}")
        console.print(f"🧊 Odinn.Orchestrator: Error saving image: {e}  ❌", style="bold red")

def create_directories(base_path: str, subfolders: List[str]) -> None:
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
        logging.debug(f"Created directory: {os.path.join(base_path, folder)}")

def save_page_data(
    out_folder: str,
    base_name: str,
    page_num: int,
    full_text: str,
    images: Dict[str, Image.Image],
    out_metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """Saves page data (text, metadata, images) to appropriate subfolders."""
    subfolder_path = get_subfolder_path(out_folder, base_name)
    create_directories(subfolder_path, ["ocr", "cleaned_text", "image"])
    text_filename = os.path.join(subfolder_path, "ocr", f"{base_name}_pg{page_num}.txt")
    save_file(full_text, text_filename)
    logging.debug(f"Saved text to {text_filename}")
    save_file(json.dumps(out_metadata, indent=4), text_filename.rsplit(".", 1)[0] + "_meta.json")
    logging.debug(f"Saved metadata to {text_filename.rsplit('.', 1)[0] + '_meta.json'}")
    for filename, image in images.items():
        save_image(image, os.path.join(subfolder_path, "image", filename))
        logging.debug(f"Saved image to {os.path.join(subfolder_path, 'image', filename)}")
    return subfolder_path, text_filename

class VisionCleanProcessor:
    """Processes an image and a text file by sending a vision model using litellm."""    
    def retrieve_image(self, image_path: str) -> str:
        """Retrieves an image from a path and converts it to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def retrieve_text(self, text_path: str) -> str:
        """Retrieves text from a path and returns it."""
        with open(text_path, "r", encoding='utf-8') as text_file:
            return text_file.read()
    
    def send_to_litellm(self, image_base64: str, text: str) -> str:
        """Sends an image and text to litellm and returns the response."""
        response = litellm.completion(
            model="ollama/llava-phi3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Restore this text to its original detail and turn this into utf-8 plain text: {text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
        )
        return response.choices[0].message.content if response.choices else None

def process_vision_clean(image_path: str, text_path: str) -> str:
    """Processes an image and a text file by sending a vision model using litellm."""
    processor = VisionCleanProcessor()
    image_base64 = processor.retrieve_image(image_path)
    text = processor.retrieve_text(text_path)
    return processor.send_to_litellm(image_base64, text)

# OCR Extraction
def process_pdf_md_only(filename: str, output: str, max_pages: Optional[int] = None, start_page: int = 0, langs: List[str] = None, batch_multiplier: int = 2) -> Tuple[str, List[str]]:
    """Processes the PDF, extracts data, and saves it."""
    fname = filename
    base_name = os.path.basename(fname).rsplit(".", 1)[0]

    console.print(f"\n🧊 Odinn.Orchestrator: Loading forensic models...  ♻️  \n", style="bold green")
    model_lst = load_all_models()

    doc = pdfium.PdfDocument(fname)
    total_pages = len(doc)

    # Validate start_page
    if start_page >= total_pages:
        raise ValueError(f"Start page {start_page + 1} is out of range. The document has only {total_pages} pages.")

    # Adjust max_pages if necessary
    if max_pages is None:
        max_pages = total_pages - start_page
    else:
        max_pages = min(max_pages, total_pages - start_page)

    ocr_text_files = []

    for page_num in range(start_page, start_page + max_pages):
        console.print(f"🧊 Odinn.Orchestrator: Processing page {page_num + 1}...", style="bold green")
        page_text, page_meta = convert_single_pdf_md_only(
            fname, model_lst, max_pages=1, langs=langs, batch_multiplier=batch_multiplier, start_page=page_num
        )

        # Extract images using pdf2image
        images = convert_from_path(fname, first_page=page_num + 1, last_page=page_num + 1)

        # Create necessary directories
        image_folder = os.path.join(output, base_name, "image")
        os.makedirs(image_folder, exist_ok=True)

        # Save the page image with bounding boxes (we are assuming the image with bounding boxes is the first one in the images list)
        image_filename = f"{base_name}_p{page_num + 1}.jpg"
        save_image(images[0], os.path.join(image_folder, image_filename))

        # Save the rest of the page data
        subfolder_path, text_filename = save_page_data(
            output, base_name, page_num + 1, page_text, {}, page_meta  # Pass an empty dictionary for images
        )
        ocr_text_files.append(text_filename)
        console.print(f"\n🧊 Odinn.Orchestrator: Saved markdown for page {page_num + 1} to the {subfolder_path} folder.    \n", style="bold blue")

        # Vision Clean process
        cleaned_text = process_vision_clean(os.path.join(image_folder, image_filename), text_filename)
        cleaned_text_filename = os.path.join(subfolder_path, "cleaned_text", f"{base_name}_pg{page_num + 1}_cleaned.txt")
        save_file(cleaned_text, cleaned_text_filename)
        console.print(f"\n🧊 Odinn.Orchestrator: Cleaned text for page {page_num + 1} saved to {cleaned_text_filename}.    \n", style="bold green")

    console.print(f"\n🧊 Odinn.Orchestrator: All pages processed.  👏  \n", style="bold green")
    return base_name, ocr_text_files

def validate_integer_input(prompt: str, error_message: str, min_value: int = None, default_value: Optional[int] = None) -> Optional[int]:
    """Prompts the user for integer input with validation. Used for user input validation."""
    while True:
        try:
            value = Prompt.ask(prompt, default=str(default_value) if default_value is not None else None)
            if value is None or not value.strip():  # Check if the input is blank
                return default_value
            value = int(value)
            if min_value is not None and value < min_value:
                raise ValueError(error_message)
            return value
        except ValueError:
            console.print(error_message, style="bold red")

def validate_string_input(prompt: str, default_value: Optional[str] = None) -> str:
    """Prompts the user for string input with a default value."""
    return Prompt.ask(prompt, default=default_value)

# Main Loop
def main():
    os.system("cls" if os.name == "nt" else "clear")
    console.print(f"🧊 Odinn.Orchestrator: PDF to Markdown Converter  \n", style="bold green")

    # Default values
    default_output_folder = "in_process"
    default_filename = "D:\\Coding\\Builds\\AIF5\\data\\Ordinance_9022\\ordinance_9022.pdf"
    default_start_page = 1
    default_max_pages = None
    default_langs = None
    default_batch_multiplier = 2

    # Prompt for user confirmation or modifications
    console.print(f"\nCurrent Settings (modify if needed):")
    console.print(f"  PDF File:           {default_filename}")
    console.print(f"  Output Folder:      {default_output_folder}")
    console.print(f"  Start Page:         {default_start_page}")
    console.print(f"  Max Pages:          {default_max_pages if default_max_pages is not None else 'All'}")
    console.print(f"  Languages:          {default_langs if default_langs is not None else 'Default'}")
    console.print(f"  Batch Multiplier:   {default_batch_multiplier}\n")

    if not Confirm.ask("Use these settings?"):
        filename = validate_string_input("Enter the PDF file to parse: ", default_value=default_filename)
        output_folder = validate_string_input("Enter the output base folder path: ", default_value=default_output_folder)
        start_page = validate_integer_input(
            "Enter the page to start processing at (leave blank for the first page): ",
            "Invalid input. Please enter a positive integer or leave blank.",
            min_value=1,
            default_value=default_start_page
        ) - 1  # Adjust for zero-indexing ONLY HERE
        max_pages = validate_integer_input(
            "Enter the maximum number of pages to parse (leave blank for all): ",
            "Invalid input. Please enter a positive integer or leave blank.",
            default_value=default_max_pages
        )
        langs = validate_string_input("Enter the languages to use for OCR, comma-separated (leave blank for default): ", default_value=default_langs)
        batch_multiplier = validate_integer_input(
            "Enter the batch multiplier: ",
            "Invalid input. Please enter a positive integer.",
            min_value=1,
            default_value=default_batch_multiplier
        )
    else:
        filename = default_filename
        output_folder = default_output_folder
        start_page = default_start_page - 1  # Adjust for zero-indexing
        max_pages = default_max_pages
        langs = default_langs
        batch_multiplier = default_batch_multiplier

    langs = langs.split(",") if langs else None

    # Run the process
    base_name, ocr_text_files = process_pdf_md_only(filename, output_folder, max_pages, start_page, langs, batch_multiplier)

if __name__ == "__main__":
    main()
