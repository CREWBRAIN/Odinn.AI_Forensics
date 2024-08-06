# 🧊 Odinn AI Forensics Tool / utils.py

import os
import shutil
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import pypdfium2 as pdfium
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, track
from rich.table import Table
from rich.panel import Panel
from imports.convert import convert_single_pdf

from settings import Settings

console = Console()

# --- City Council Metadata ---
MANILA_CITY_COUNCIL_DATA = {
    12: {"ordinance_range": (8883, None), "term": "July 01, 2022 - Present"},
    11: {"ordinance_range": (8564, 8882), "term": "July 01, 2019 - June 30, 2022"},
    10: {"ordinance_range": (8498, 8563), "term": "July 01, 2016 - June 30, 2019"},
    9: {"ordinance_range": (8323, 8497), "term": "July 01, 2013 - June 30, 2016"},
    8: {"ordinance_range": (8224, 8322), "term": "July 01, 2010 - June 30, 2013"},
    7: {"ordinance_range": (8141, 8223), "term": "July 01, 2007 - June 30, 2010"},
    6: {"ordinance_range": (8078, 8140), "term": "July 01, 2004 - June 30, 2007"},
    5: {"ordinance_range": (8024, 8077), "term": "July 01, 2001 - June 30, 2004"},
    4: {"ordinance_range": (7955, 8024), "term": "July 01, 1998 - June 30, 2001"},
    3: {"ordinance_range": (7890, 7954), "term": "July 01, 1995 - June 30, 1998"},
    2: {"ordinance_range": (7766, 7889), "term": "July 01, 1992 - June 30, 1995"},
    1: {"ordinance_range": (7682, 7765), "term": "February 02, 1988 - June 30, 1992"},
    0: {"ordinance_range": (2926, 7681), "term": "1945 - 1975"}  # Municipal Board
}

def create_directories(base_path: str, subfolders: List[str]) -> None:
    """Creates directories if they don't exist."""
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
        logging.debug(f"Created directory: {os.path.join(base_path, folder)}")

def save_file(content: str, path: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """Saves text content to a file."""
    with open(path, mode, encoding=encoding) as f:
        f.write(content)

def save_image(image: Image.Image, path: str) -> None:
    """Saves an image to a file."""
    try:
        image.save(path, "JPEG")
        logging.debug(f"Saved image to {path}")
    except Exception as e:
        logging.error(f"Error saving image to {path}: {e}")
        console.print(f"🧊 Odinn.Orchestrator: Error saving image: {e}  ❌", style="bold red")

def extract_image_with_bboxes(page: pdfium.PdfPage, bboxes: List[Dict], settings: Settings, output_path: str) -> None:
    """Extracts a page image and visualizes bounding boxes."""
    # Use pdf2image to extract the image
    images = convert_from_path(page, first_page=1, last_page=1)
    image = images[0]

    # Draw bounding boxes if enabled
    if settings.visualize_bboxes:
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            # Filter by bounding box type if specified
            if settings.bbox_types and bbox["label"] not in settings.bbox_types:
                continue
            coords = bbox["coords"]  # Assuming coords are in the format [x0, y0, x1, y1]
            draw.rectangle(coords, outline="red", width=2)  # Customize color and width as needed

    # Save the image
    image.save(output_path, "JPEG")

def get_subfolder_path(out_folder: str, fname: str) -> str:
    base_name = os.path.basename(fname).rsplit(".", 1)[0]
    return os.path.join(out_folder, base_name)

# --- State Management ---
def save_state(state: Dict, filepath: str) -> None:
    """Saves the processing state to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=4)

def load_state(filepath: str) -> Optional[Dict]:
    """Loads the processing state from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            state = json.load(f)
        return state
    except FileNotFoundError:
        console.print(f"[bold red]Error: State file not found: {filepath}[/]")
        return None
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Invalid JSON in state file: {filepath}[/]")
        return None

# --- Metadata Functions ---
def get_user_metadata() -> Dict:
    """Prompts the user for metadata input."""
    user_metadata = {}
    user_metadata["title"] = Prompt.ask("Enter ordinance title:")
    
    # Use MANILA_CITY_COUNCIL_DATA for council selection
    council_choices = [str(i) for i in MANILA_CITY_COUNCIL_DATA.keys()]
    user_metadata["council"] = int(Prompt.ask("Enter council number", choices=council_choices))
    
    ordinance_number = IntPrompt.ask("Enter ordinance number:")
    user_metadata["ordinance_number"] = ordinance_number

    # Validate ordinance number against council data
    council_data = MANILA_CITY_COUNCIL_DATA[user_metadata["council"]]
    start, end = council_data["ordinance_range"]
    if start <= ordinance_number and (end is None or ordinance_number <= end):
        console.print(f"Ordinance {ordinance_number} belongs to council {user_metadata['council']}")
        console.print(f"Term: {council_data['term']}")
    else:
        console.print("[bold yellow]Warning: Ordinance number does not match the selected council.[/]")

    user_metadata["year"] = IntPrompt.ask("Enter year:")
    
    return user_metadata

def extract_metadata_from_text(text: str) -> Dict:
    """Extracts metadata from cleaned text using regex."""
    extracted_metadata = {}
    import re
    match = re.search(r'Ordinance No\. (\d+)', text)
    if match:
        ordinance_number = int(match.group(1))  # Convert to integer

        # Determine City Council based on ordinance number
        for council, data in MANILA_CITY_COUNCIL_DATA.items():
            start, end = data["ordinance_range"]
            if start <= ordinance_number and (end is None or ordinance_number <= end):
                extracted_metadata["council"] = council
                extracted_metadata["term"] = data["term"]
                break

    # Extract year (assuming it's mentioned in the text)
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if year_match:
        extracted_metadata["year"] = int(year_match.group(1))

    # ... (Add more regex patterns to extract other metadata)
    return extracted_metadata

def merge_metadata(user_metadata: Dict, extracted_metadata: Dict) -> Dict:
    """Merges user-provided and extracted metadata."""
    merged_metadata = user_metadata.copy()
    for key, value in extracted_metadata.items():
        if key not in merged_metadata or not merged_metadata[key]:  # Prioritize user input
            merged_metadata[key] = value
    return merged_metadata

# --- Main Processing Functions ---
def process_page(doc: pdfium.PdfDocument, page_num: int, settings: Settings, output_folder: str) -> Tuple[str, Dict, Dict[str, Image.Image]]:
    """Processes a single page of the PDF."""
    console.print(f"🧊 Odinn.Orchestrator: Processing page {page_num + 1}...", style="bold green")
    
    # Create necessary directories
    image_folder = os.path.join(output_folder, "page_images")
    image_with_bboxes_folder = os.path.join(output_folder, "page_images_bboxes")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(image_with_bboxes_folder, exist_ok=True)

    # Extract the page from the PdfDocument
    page = doc[page_num]
    
    # Convert the page to an image
    pil_image = page.render().to_pil()
    
    # Save the original image
    image_filename = f"page_{page_num + 1}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    save_image(pil_image, image_path)
    
    # Use marker's convert_single_pdf with the image path
    page_text, images, page_meta = convert_single_pdf(
        image_path,
        model_lst=None,
        max_pages=1,
        langs=settings.langs,
        batch_multiplier=settings.batch_multiplier,
        start_page=0  # We're processing a single image, so start_page is always 0
    )
    
    # Extract bounding boxes if enabled
    bboxes = []
    if settings.extract_bboxes:
        bboxes = page_meta.get('detected_bboxes', [])

    # Visualize bounding boxes on the image
    if settings.visualize_bboxes:
        image_with_bboxes = pil_image.copy()
        draw = ImageDraw.Draw(image_with_bboxes)
        for bbox in bboxes:
            if settings.bbox_types and bbox["label"] not in settings.bbox_types:
                continue
            coords = bbox["coords"]
            draw.rectangle(coords, outline="red", width=2)
        
        # Save the image with bounding boxes
        image_with_bboxes_filename = f"page_{page_num + 1}_bboxes.jpg"
        image_with_bboxes_path = os.path.join(image_with_bboxes_folder, image_with_bboxes_filename)
        save_image(image_with_bboxes, image_with_bboxes_path)

    console.print(f"🧊 Odinn.Orchestrator: Page {page_num + 1} processed.", style="bold green")
    return page_text, page_meta, images

def load_settings_from_file(filepath: str) -> Settings:
    """Loads settings from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        settings = Settings()
        settings.load_from_dict(data)
        return settings
    except FileNotFoundError:
        console.print(f"[bold red]Error: Settings file not found: {filepath}[/]")
        return Settings()  # Return default settings if file not found
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Invalid JSON in settings file: {filepath}[/]")
        return Settings()  # Return default settings if JSON is invalid

def display_bounding_box_options(settings: Settings):
    """Displays the bounding box options menu."""
    while True:
        console.print(Panel(
            f"[bold blue]Bounding Box Options:[/]\n"
            f"Extract Bounding Boxes: {settings.extract_bboxes}\n"
            f"Bounding Box Types: {', '.join(settings.bbox_types) if settings.bbox_types else 'All'}\n"
            f"Visualize Bounding Boxes: {settings.visualize_bboxes}",
            title="Bounding Box Options",
            expand=False
        ))

        options = [
            "Toggle Extract Bounding Boxes",
            "Select Bounding Box Types",
            "Toggle Visualize Bounding Boxes",
            "Back to Settings"
        ]
        choice = Prompt.ask("Select an option", choices=[str(i+1) for i in range(len(options))], default=str(len(options)))

        if choice == str(len(options)):
            break  # Back to settings menu

        choice_index = int(choice) - 1
        option = options[choice_index]

        if option == "Toggle Extract Bounding Boxes":
            settings.extract_bboxes = not settings.extract_bboxes
        elif option == "Select Bounding Box Types":
            available_types = ["Table", "Figure", "Formula", "Section-Header", "Title", "List-Item", "Code"]  # Get from marker.schema.bbox
            selected_types = []
            for bbox_type in available_types:
                if Confirm.ask(f"Include {bbox_type}?", default=False):
                    selected_types.append(bbox_type)
            settings.bbox_types = selected_types if selected_types else None
        elif option == "Toggle Visualize Bounding Boxes":
            settings.visualize_bboxes = not settings.visualize_bboxes