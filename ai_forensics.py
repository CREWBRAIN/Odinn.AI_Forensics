# 🧊 Odinn AI Forensics Tool / ai_forensics.py

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

# Import from local marker package
from imports.convert import convert_single_pdf
from imports.extract_text import get_toc
from marker.settings import settings as marker_settings
from imports.models import load_all_models

# Import project modules
from utils import (
    create_directories, save_file, save_image, extract_image_with_bboxes, get_subfolder_path,
    save_state, load_state, get_user_metadata, extract_metadata_from_text, merge_metadata, process_page,
    load_settings_from_file, MANILA_CITY_COUNCIL_DATA
)
from settings import Settings
from vision_cleaning import VisionCleanProcessor, process_vision_clean

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set log level to DEBUG for verbose logging
console = Console()

# --- Main Processing Functions ---
def debug_print(message):
    logging.debug(message)

# In your main processing functions, replace progress bars with debug prints
def process_pdf(filename, output_folder, settings, user_metadata=None):
    debug_print(f"Starting to process PDF: {filename}")
    
    # Create project folder
    project_folder = get_subfolder_path(output_folder, filename)
    create_directories(project_folder, ["original_pdf", "page_images", "page_images_bboxes", "page_data"])
    debug_print(f"Created project folder: {project_folder}")

    # Copy original PDF
    shutil.copy(filename, os.path.join(project_folder, "original_pdf", os.path.basename(filename)))
    debug_print("Copied original PDF")

    # Load PDF document
    doc = pdfium.PdfDocument(filename)
    total_pages = len(doc)
    debug_print(f"Loaded PDF with {total_pages} pages")

    # Process pages
    all_page_text = []
    all_page_metadata = []
    for page_num in range(total_pages):
        debug_print(f"Processing page {page_num + 1}")
        page_text, page_meta, images = process_page(doc, page_num, settings, project_folder)
        all_page_text.append(page_text)
        all_page_metadata.append(page_meta)

        # Save page data
        save_file(page_text, os.path.join(project_folder, "page_data", f"page_{page_num + 1}_text.txt"))
        save_file(json.dumps(page_meta, indent=4), os.path.join(project_folder, "page_data", f"page_{page_num + 1}_metadata.json"))
        debug_print(f"Saved data for page {page_num + 1}")

    # Extract metadata
    debug_print("Extracting metadata")
    all_text = "\n".join(all_page_text)
    extracted_metadata = extract_metadata_from_text(all_text)
    merged_metadata = merge_metadata(user_metadata or {}, extracted_metadata)

    # Save merged metadata
    save_file(json.dumps(merged_metadata, indent=4), os.path.join(project_folder, "metadata.json"))
    debug_print("Saved merged metadata")

    # Extract TOC
    debug_print("Extracting table of contents")
    toc = get_toc(doc)
    save_file(json.dumps(toc, indent=4), os.path.join(project_folder, "page_data", "toc.json"))

    debug_print(f"Processing complete for {filename}")
    debug_print(f"Project folder: {project_folder}")
    debug_print(f"Total pages processed: {total_pages}")

# Similarly, update other functions that use rich progress bars or live displays

# --- TUI Functions ---
def display_main_menu():
    """Displays the main menu options."""
    table = Table(title="🧊 Odinn AI Forensics Tool")
    table.add_column("Option", style="cyan", width=12)
    table.add_column("Description", style="magenta")
    table.add_row("[1]", "Load PDF")
    table.add_row("[2]", "Settings")
    table.add_row("[3]", "Process PDF (Step-by-Step)")
    table.add_row("[4]", "Process PDF (Uber Step)")
    table.add_row("[5]", "Load Saved State")
    table.add_row("[6]", "Exit")
    console.print(table)

def display_settings_menu(settings: Settings):
    """Displays the settings menu."""
    console.print(Panel(f"[bold blue]Current Settings:[/]\n{settings}", title="Settings", expand=False))

    while True:
        options = [
            "Torch Device",
            "OCR Engine",
            "Languages",
            "Batch Multiplier",
            "Bounding Box Options",
            "Output Folder",
            "Citation Information",
            "Save Settings",
            "Load Settings",
            "Back to Main Menu"
        ]
        choice = Prompt.ask("Select an option", choices=[str(i+1) for i in range(len(options))], default=str(len(options)))

        if choice == str(len(options)):
            break  # Back to main menu

        choice_index = int(choice) - 1
        option = options[choice_index]

        if option == "Torch Device":
            devices = ["cpu", "cuda", "mps"]
            device_choice = Prompt.ask("Select Torch device", choices=devices, default=settings.torch_device)
            settings.torch_device = device_choice
        elif option == "OCR Engine":
            engines = ["surya", "ocrmypdf", "None"]
            engine_choice = Prompt.ask("Select OCR engine", choices=engines, default=settings.ocr_engine)
            settings.ocr_engine = engine_choice if engine_choice != "None" else None
        elif option == "Languages":
            langs_str = Prompt.ask("Enter languages (comma-separated)", default=", ".join(settings.langs) if settings.langs else "")
            settings.langs = [lang.strip() for lang in langs_str.split(",")] if langs_str else None
        elif option == "Batch Multiplier":
            settings.batch_multiplier = IntPrompt.ask("Enter batch multiplier", default=settings.batch_multiplier)
        elif option == "Bounding Box Options":
            display_bounding_box_options(settings)
        elif option == "Output Folder":
            settings.output_folder = Prompt.ask("Enter output folder path", default=settings.output_folder)
        elif option == "Citation Information":
            settings.citation["title"] = Prompt.ask("Enter document title:", default=settings.citation.get("title", ""))
            settings.citation["author"] = Prompt.ask("Enter document author:", default=settings.citation.get("author", ""))
            # ... (Prompt for other citation fields)
        elif option == "Save Settings":
            filepath = Prompt.ask("Enter settings file path to save:")
            settings.save_to_file(filepath)
        elif option == "Load Settings":
            filepath = Prompt.ask("Enter settings file path to load:")
            loaded_settings = load_settings_from_file(filepath)
            if loaded_settings:
                settings = loaded_settings

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

def display_step_by_step_menu(filename: str, output_folder: str, settings: Settings, user_metadata: Optional[Dict] = None):
    """Displays the step-by-step processing menu."""
    doc = pdfium.PdfDocument(filename)
    total_pages = len(doc)

    while True:
        table = Table(title="Step-by-Step Processing")
        table.add_column("Option", style="cyan", width=12)
        table.add_column("Description", style="magenta")
        table.add_row("[1]", "Extract Text and Metadata (Marker)")
        table.add_row("[2]", "Extract Images")
        table.add_row("[3]", "Vision Cleaning")
        table.add_row("[4]", "Generate Output Package")
        table.add_row("[5]", "Save Current State")
        table.add_row("[6]", "Back to Main Menu")
        console.print(table)

        choice = Prompt.ask("Enter option", choices=["1", "2", "3", "4", "5", "6"], default="6")

        if choice == "1":
            # IMPLEMENTED: Extracting text and metadata
            print("[cyan]Extracting text and metadata...")  # REPLACED: Rich progress with a simple print statement
            for page_num in range(total_pages):
                process_page(doc, page_num, settings, output_folder)
                print(f"Processed page {page_num + 1}/{total_pages}")  # REPLACED: Progress update with a print statement
        elif choice == "2":
            display_image_extraction_menu(doc, settings, output_folder)
        elif choice == "3":
            # IMPLEMENTED: Vision cleaning for selected pages
            page_range = Prompt.ask("Enter page range (e.g., 1-5, or 'all' for all pages)")
            if page_range.lower() == 'all':
                pages_to_clean = range(total_pages)
            else:
                start, end = map(int, page_range.split('-'))
                pages_to_clean = range(start - 1, end)
            
            print("[cyan]Performing vision cleaning...")  # REPLACED: Rich progress with a simple print statement
            for page_num in pages_to_clean:
                page_text, page_meta, images = process_page(doc, page_num, settings, output_folder)
                cleaned_text = process_vision_clean(
                    os.path.join(output_folder, "page_images_bboxes", f"page_{page_num + 1}_bboxes.jpg"),
                    os.path.join(output_folder, "page_data", f"page_{page_num + 1}_text.txt"),
                    page_meta,
                    settings
                )
                save_file(cleaned_text, os.path.join(output_folder, "page_data", f"page_{page_num + 1}_cleaned_text.txt"))
                print(f"Cleaned page {page_num + 1}/{len(pages_to_clean)}")  # REPLACED: Progress update with a print statement
        elif choice == "4":
            # IMPLEMENTED: Output package generation
            generate_output_package(filename, output_folder, settings, user_metadata)
        elif choice == "5":
            # IMPLEMENTED: State saving
            state = {
                "filename": filename,
                "output_folder": output_folder,
                "settings": settings.__dict__,
                "user_metadata": user_metadata
            }
            state_filepath = Prompt.ask("Enter state file path to save:")
            save_state(state, state_filepath)
            console.print(f"State saved to: {state_filepath}")
        elif choice == "6":
            break  # Back to main menu

    while True:
        table = Table(title="Step-by-Step Processing")
        table.add_column("Option", style="cyan", width=12)
        table.add_column("Description", style="magenta")
        table.add_row("[1]", "Extract Text and Metadata (Marker)")
        table.add_row("[2]", "Extract Images")
        table.add_row("[3]", "Vision Cleaning")
        table.add_row("[4]", "Generate Output Package")
        table.add_row("[5]", "Save Current State")
        table.add_row("[6]", "Back to Main Menu")
        console.print(table)

        choice = Prompt.ask("Enter option", choices=["1", "2", "3", "4", "5", "6"], default="6")

        if choice == "1":
            print("[cyan]Extracting text and metadata...")  # REPLACED: Rich progress with a simple print statement
            for page_num in range(total_pages):
                process_page(doc, page_num, settings, output_folder)
                print(f"Processed page {page_num + 1}/{total_pages}")  # REPLACED: Progress update with a print statement
        elif choice == "2":
            display_image_extraction_menu(doc, settings, output_folder)
        elif choice == "3":
            # IMPLEMENTED: Vision cleaning for selected pages
            page_range = Prompt.ask("Enter page range (e.g., 1-5, or 'all' for all pages)")
            if page_range.lower() == 'all':
                pages_to_clean = range(total_pages)
            else:
                start, end = map(int, page_range.split('-'))
                pages_to_clean = range(start - 1, end)
            
            print("[cyan]Performing vision cleaning...")  # REPLACED: Rich progress with a simple print statement
            for page_num in pages_to_clean:
                page_text, page_meta, images = process_page(doc, page_num, settings, output_folder)
                cleaned_text = process_vision_clean(
                    os.path.join(output_folder, "page_images_bboxes", f"page_{page_num + 1}_bboxes.jpg"),
                    os.path.join(output_folder, "page_data", f"page_{page_num + 1}_text.txt"),
                    page_meta,
                    settings
                )
                save_file(cleaned_text, os.path.join(output_folder, "page_data", f"page_{page_num + 1}_cleaned_text.txt"))
                print(f"Cleaned page {page_num + 1}/{len(pages_to_clean)}")  # REPLACED: Progress update with a print statement
        elif choice == "4":
            # IMPLEMENTED: Output package generation
            generate_output_package(filename, output_folder, settings, user_metadata)
        elif choice == "5":
            # IMPLEMENTED: State saving
            state = {
                "filename": filename,
                "output_folder": output_folder,
                "settings": settings.__dict__,
                "user_metadata": user_metadata
            }
            state_filepath = Prompt.ask("Enter state file path to save:")
            save_state(state, state_filepath)
            console.print(f"State saved to: {state_filepath}")
        elif choice == "6":
            break  # Back to main menu

def display_image_extraction_menu(doc: pdfium.PdfDocument, settings: Settings, output_folder: str):
    """Displays the image extraction options menu."""
    total_pages = len(doc)

    while True:
        table = Table(title="Image Extraction")
        table.add_column("Option", style="cyan", width=12)
        table.add_column("Description", style="magenta")
        table.add_row("[1]", "Extract All Images (No Bounding Boxes)")
        table.add_row("[2]", "Extract All Images (With Bounding Boxes)")
        table.add_row("[3]", "Back to Step-by-Step Menu")
        console.print(table)

        choice = Prompt.ask("Enter option", choices=["1", "2", "3"], default="3")

        if choice == "1":
            with Progress() as progress:
                for page_num in track(range(total_pages), description="Extracting images..."):
                    _, _, images = process_page(doc, page_num, settings, output_folder)
                    # IMPLEMENTED: Save images without bounding boxes
                    for idx, img in enumerate(images):
                        save_image(img, os.path.join(output_folder, "page_images", f"page_{page_num + 1}_image_{idx + 1}.jpg"))
        elif choice == "2":
            with Progress() as progress:
                for page_num in track(range(total_pages), description="Extracting images with bounding boxes..."):
                    _, page_meta, images = process_page(doc, page_num, settings, output_folder)
                    # IMPLEMENTED: Save images with bounding boxes
                    bboxes = page_meta.get('detected_bboxes', [])
                    extract_image_with_bboxes(doc[page_num], bboxes, settings, os.path.join(output_folder, "page_images_bboxes", f"page_{page_num + 1}_bboxes.jpg"))
        elif choice == "3":
            break  # Back to step-by-step menu

def generate_output_package(filename: str, output_folder: str, settings: Settings, user_metadata: Optional[Dict] = None):
    """Generates the final output package."""
    console.print("🧊 Odinn.Orchestrator: Generating output package...", style="bold green")
    
    # Create a summary markdown file
    summary_md = f"# 🧊 Odinn AI Forensics Tool - Summary Report\n\n"
    summary_md += f"## Document Information\n"
    summary_md += f"- Filename: {os.path.basename(filename)}\n"
    summary_md += f"- Output Folder: {output_folder}\n\n"
    
    # Add metadata
    summary_md += f"## Metadata\n"
    metadata_file = os.path.join(output_folder, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        for key, value in metadata.items():
            summary_md += f"- {key}: {value}\n"
    
    # Add TOC
    toc_file = os.path.join(output_folder, "page_data", "toc.json")
    if os.path.exists(toc_file):
        with open(toc_file, 'r') as f:
            toc = json.load(f)
        summary_md += "## Table of Contents\n"
        for item in toc:
            summary_md += f"- {item['title']}\n"
    
    # Add information about extracted images
    image_folder = os.path.join(output_folder, "page_images")
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    summary_md += f"\n## Extracted Images\n"
    summary_md += f"Total images extracted: {len(image_files)}\n"

    # Add information about cleaned text
    cleaned_text_folder = os.path.join(output_folder, "page_data")
    cleaned_text_files = [f for f in os.listdir(cleaned_text_folder) if f.endswith('_cleaned_text.txt')]
    summary_md += f"\n## Cleaned Text\n"
    summary_md += f"Pages with cleaned text: {len(cleaned_text_files)}\n"

    # Save summary report
    save_file(summary_md, os.path.join(output_folder, "summary_report.md"))

    console.print("🧊 Odinn.Orchestrator: Output package generated successfully!", style="bold green")

# --- Main Function ---
def main():
    """Main function to run the TUI."""
    settings = Settings()
    user_metadata = None
    filename = None
    model_lst = None

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        display_main_menu()
        choice = Prompt.ask("Enter option", choices=["1", "2", "3", "4", "5", "6"], default="1")

        if choice == "1":
            filename = Prompt.ask("Enter PDF file path")
            if not os.path.exists(filename):
                console.print(f"[bold red]Error: File not found: {filename}[/]")
                filename = None
            else:
                console.print(f"Loaded PDF: {filename}")
                # Set the output folder based on the input file
                output_folder = os.path.join("output", os.path.splitext(os.path.basename(filename))[0])
                settings.set_output_folder(output_folder)
                Prompt.ask("Press Enter to continue")  # Add this line to pause
        elif choice == "2":
            display_settings_menu(settings)
        elif choice == "3":
            if filename:
                display_step_by_step_menu(filename, settings.output_folder, settings, user_metadata)
            else:
                console.print("[bold red]Error: No PDF loaded.[/]")
                Prompt.ask("Press Enter to continue")  # Add this line to pause
        elif choice == "4":
            if filename:
                user_metadata = get_user_metadata()
                process_pdf(filename, settings.output_folder, settings, user_metadata)
                Prompt.ask("Press Enter to continue")  # Add this line to pause
            else:
                console.print("[bold red]Error: No PDF loaded.[/]")
                Prompt.ask("Press Enter to continue")  # Add this line to pause
        elif choice == "5":
            state_filepath = Prompt.ask("Enter state file path to load")
            state = load_state(state_filepath)
            if state:
                # ... (Load settings, filename, user_metadata, and other state information)
                console.print(f"Loaded state from: {state_filepath}")
                Prompt.ask("Press Enter to continue")  # Add this line to pause
        elif choice == "6":
            console.print("Exiting 🧊 Odinn AI Forensics Tool...")
            break

if __name__ == "__main__":
    main()