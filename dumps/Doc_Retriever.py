# Doc_Retriever_4.3_6.py
# Working Wonderfully with API Support

import os
import datetime
import logging
import tiktoken
import yaml
import shutil
import re
import base64

from tqdm import tqdm
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rich.table import Table

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# --- Configuration ---
config = {
    "Settings": {
        "lgu": "Your LGU Name",
        "DEFAULT_INPUT_FOLDER": r"D:\Coding\Builds\AIF5\reference",
        "DEFAULT_OUTPUT_FOLDER": r"D:\Coding\Builds\AIF5\output"
    },
    "Templates": {
        "header": "LGU: {lgu}\nDocument Type: {document_type}\nDocument Number: {document_number}\nPage: {page_number} of {page_totals}\nOriginal File: {original_file_location}\nTimestamp: {timestamp}",
        "footer": "End of document. Processed by {lgu} AI-Forensics File Processor."
    }
}

# --- Logging Setup ---
logging.basicConfig(filename="AI_Forensics_File_Processor.log", level=logging.INFO)

# Add this after the config and logging setup:
app = FastAPI(title="AI-Forensics File Processor API", version="4.3.6")

# Add these API models:
class DigestRequest(BaseModel):
    start_doc_num: int
    end_doc_num: int
    max_tokens: int

class DocumentTextRequest(BaseModel):
    document_number: str

class ImageRetrievalRequest(BaseModel):
    ordinance_number: str

class ImageResponse(BaseModel):
    filename: str
    content: str  # Base64 encoded image content

# --- Helper Functions ---
def extract_info_from_filename(filename):
    try:
        parts = filename.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid file name format: {filename}")
        document_type = parts[0]
        document_number = parts[1]
        page_part = parts[2]
        page_number = int(page_part.split(".")[0].replace("pg", ""))
        return document_type, document_number, page_number
    except (ValueError, IndexError) as e:
        logging.error(f"Error extracting info from filename {filename}: {e}")
        return None, None, None

def generate_header(document_type, document_number, page_number, original_file_location, page_totals):
    header_template = config["Templates"]["header"]
    timestamp = datetime.datetime.now().isoformat()
    header = header_template.format(
        lgu=config["Settings"]["lgu"],
        document_type=document_type,
        document_number=document_number,
        page_number=page_number,
        page_totals=page_totals,
        original_file_location=original_file_location,
        timestamp=timestamp
    )
    return header

def generate_footer():
    footer_template = config["Templates"]["footer"]
    footer = footer_template.format(lgu=config["Settings"]["lgu"])
    return footer

def process_file(file_path, document_number, document_type, page_number, page_totals, output_folder):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        header = generate_header(document_type, document_number, page_number, file_path, page_totals)
        footer = generate_footer()
        updated_content = f"{header}\n\n{content}\n\n{footer}"

        output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)

        logging.info(f"Processed file: {file_path} -> {output_file_path}")
    except Exception as error:
        logging.error(f"Error processing file {file_path}: {error}")

# --- Process Directory Function ---
def process_directory():
    console = Console()
    console.print(Panel("[bold blue]AI-Forensics File Processor - Process Directory[/bold blue]"))

    input_folder = config["Settings"]["DEFAULT_INPUT_FOLDER"]
    output_folder = config["Settings"]["DEFAULT_OUTPUT_FOLDER"]

    # Create folders if they don't exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    file_paths = []
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_paths.append(os.path.join(root, filename))

    logging.info(f"Found {len(file_paths)} files to process in {input_folder}")
    console.print(f"[green]Found {len(file_paths)} files to process.[/green]")

    document_pages = {}
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        document_type, document_number, page_number = extract_info_from_filename(file_name)
        if document_number not in document_pages:
            document_pages[document_number] = []
        document_pages[document_number].append(page_number)

    with console.status("[bold green]Processing files...") as status:
        with tqdm(total=len(file_paths), desc="Processing", unit=" files") as progress_bar:
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                document_type, document_number, page_number = extract_info_from_filename(file_name)
                page_totals = len(document_pages[document_number])
                process_file(file_path, document_number, document_type, page_number, page_totals, output_folder)
                progress_bar.update(1)

    console.print(Panel("[bold green]Directory processed successfully.[/bold green]"))

# --- Sort Output Files Function ---
def sort_output_files():
    console = Console()
    console.print(Panel("[bold blue]AI-Forensics File Processor - Sort Output Files[/bold blue]"))

    reference_folder = config["Settings"]["DEFAULT_INPUT_FOLDER"]
    output_folder = config["Settings"]["DEFAULT_OUTPUT_FOLDER"]

    # Create folders if they don't exist
    os.makedirs(reference_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Prepare to track all files for progress reporting
    all_files = []
    for root, dirs, files in os.walk(reference_folder):
        for file in files:
            if file.endswith((".txt", ".png", ".jpg", ".pdf")):
                all_files.append(os.path.join(root, file))

    # Copy files from reference folder to output folder with progress meter
    with tqdm(total=len(all_files), desc="Copying files", unit="file") as progress_bar:
        for file_path in all_files:
            relative_path = os.path.relpath(file_path, reference_folder)
            output_file_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            shutil.copy2(file_path, output_file_path)
            logging.info(f"Copied file: {file_path} -> {output_file_path}")
            progress_bar.update(1)

    # Sort output files based on document type and number
    sorted_files = []
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            if file.endswith(".txt"):
                sorted_files.append(os.path.join(root, file))

    with tqdm(total=len(sorted_files), desc="Sorting files", unit="file") as progress_bar:
        for file_path in sorted_files:
            document_type, document_number, page_number = extract_info_from_filename(os.path.basename(file_path))

            if document_type and document_number:
                if "_pg" in file_path:
                    output_subdir = os.path.join(output_folder, f"{document_type}_{document_number}")
                else:
                    output_subdir = os.path.join(output_folder, f"{document_type}_{document_number}_(Original)")

                os.makedirs(output_subdir, exist_ok=True)
                output_file_name = os.path.basename(file_path)
                output_file_path = os.path.join(output_subdir, output_file_name)
                shutil.move(file_path, output_file_path)
                logging.info(f"Moved file: {file_path} -> {output_file_path}")
            else:
                logging.warning(f"Skipped file due to invalid filename format: {file_path}")

            progress_bar.update(1)

    console.print(Panel("[bold green]Output files sorted successfully.[/bold green]"))

# --- Create Digest Function ---
def create_digest(start_doc_num, end_doc_num, max_tokens):
    console = Console()
    console.print(Panel("[bold blue]AI-Forensics File Processor - Create Digest[/bold blue]"))

    output_folder = config["Settings"]["DEFAULT_OUTPUT_FOLDER"]

    # Initialize tiktoken encoder
    enc = tiktoken.get_encoding("cl100k_base")

    current_batch = 1
    current_token_count = 0
    digest_content = ""
    included_ordinances = []
    created_digests = []

    for document_number in range(start_doc_num, end_doc_num + 1):
        document_folder = os.path.join(output_folder, f"Ordinance_{document_number}")
        if os.path.exists(document_folder):
            page_files = sorted([f for f in os.listdir(document_folder) if f.endswith(".txt") and "_pg" in f])
            document_content = ""
            for file_name in page_files:
                file_path = os.path.join(document_folder, file_name)
                document_type, doc_number, page_number = extract_info_from_filename(file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as page_file:
                        content = page_file.read()
                        header = generate_header(document_type, doc_number, page_number, file_path, len(page_files))
                        page_content = f"{header}\n\n{content}\n\n"
                        
                        # Check if adding this page would exceed the token limit
                        new_token_count = current_token_count + len(enc.encode(page_content))
                        if new_token_count <= max_tokens:
                            document_content += page_content
                            current_token_count = new_token_count
                        else:
                            # If we can't add the entire document, break the loop
                            break
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
            
            # If we were able to add the entire document, include it in the digest
            if document_content:
                digest_content += document_content
                included_ordinances.append(document_number)
            else:
                # If we couldn't add the document, create a new batch
                if digest_content:
                    footer = generate_footer()
                    digest_content += footer
                    
                    # Create filename with included ordinance numbers
                    digest_filename = f"ordinance_digest_{included_ordinances[0]}_{included_ordinances[-1]}_batch_{current_batch}.txt"
                    digest_filepath = os.path.join(output_folder, digest_filename)
                    
                    with open(digest_filepath, "w", encoding="utf-8") as digest_file:
                        digest_file.write(digest_content)
                    
                    logging.info(f"Digest file created: {digest_filepath}")
                    console.print(f"[green]Digest batch {current_batch} created: {digest_filepath}[/green]")
                    console.print(f"[green]Included ordinances: {included_ordinances}[/green]")
                    console.print(f"[green]Total tokens: {current_token_count}[/green]")
                    
                    created_digests.append({
                        "batch": current_batch,
                        "filename": digest_filename,
                        "included_ordinances": included_ordinances,
                        "token_count": current_token_count
                    })
                    
                    # Reset for next batch
                    current_batch += 1
                    current_token_count = 0
                    digest_content = ""
                    included_ordinances = []
                    
                    # Try to add the current document to the new batch
                    new_token_count = len(enc.encode(document_content))
                    if new_token_count <= max_tokens:
                        digest_content = document_content
                        current_token_count = new_token_count
                        included_ordinances.append(document_number)
        else:
            logging.warning(f"Document folder not found for document number {document_number}")

    # Create the final batch if there's any remaining content
    if digest_content:
        footer = generate_footer()
        digest_content += footer
        
        digest_filename = f"ordinance_digest_{included_ordinances[0]}_{included_ordinances[-1]}_batch_{current_batch}.txt"
        digest_filepath = os.path.join(output_folder, digest_filename)
        
        with open(digest_filepath, "w", encoding="utf-8") as digest_file:
            digest_file.write(digest_content)
        
        logging.info(f"Final digest file created: {digest_filepath}")
        console.print(f"[green]Final digest batch {current_batch} created: {digest_filepath}[/green]")
        console.print(f"[green]Included ordinances: {included_ordinances}[/green]")
        console.print(f"[green]Total tokens: {current_token_count}[/green]")
        
        created_digests.append({
            "batch": current_batch,
            "filename": digest_filename,
            "included_ordinances": included_ordinances,
            "token_count": current_token_count
        })

    console.print(Panel("[bold green]Digest creation process completed.[/bold green]"))
    return created_digests

# --- Retrieve Document Text Function ---
def retrieve_document_text(document_number):
    console = Console()
    console.print(Panel("[bold blue]AI-Forensics File Processor - Retrieve Document Text[/bold blue]"))

    output_folder = config["Settings"]["DEFAULT_OUTPUT_FOLDER"]

    document_folder = os.path.join(output_folder, f"Ordinance_{document_number}")
    if not os.path.exists(document_folder):
        console.print(f"[bold red]Document {document_number} not found.[/bold red]")
        return None

    document_text = ""
    page_files = sorted([f for f in os.listdir(document_folder) if f.endswith(".txt") and "_pg" in f])
    total_pages = len(page_files)

    console.print(f"[green]Retrieving document {document_number} with {total_pages} pages...[/green]")

    for file_name in page_files:
        file_path = os.path.join(document_folder, file_name)
        document_type, doc_number, page_number = extract_info_from_filename(file_name)
        try:
            with open(file_path, "r", encoding="utf-8", errors='replace') as page_file:
                content = page_file.read()
                header = generate_header(document_type, doc_number, page_number, file_path, total_pages)
                document_text += f"{header}\n\n{content}\n\n"
            console.print(f"[green]Processed page {page_number} of {total_pages}[/green]")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            console.print(f"[bold red]Error reading file {file_path}: {e}[/bold red]")

    footer = generate_footer()
    document_text += footer

    # Save the retrieved document text
    output_filename = os.path.join(output_folder, f"Retrieved_Document_{document_number}.txt")
    with open(output_filename, "w", encoding="utf-8") as output_file:
        output_file.write(document_text)

    console.print(f"\n[bold green]Document text retrieved and saved as: {output_filename}[/bold green]")
    return document_text

# --- Retrieve Images Function ---
def retrieve_images(ordinance_number):
    console = Console()
    console.print(Panel("[bold blue]AI-Forensics File Processor - Retrieve Images[/bold blue]"))

    output_folder = config["Settings"]["DEFAULT_OUTPUT_FOLDER"]

    image_files = []
    search_pattern = f"Ordinance_{ordinance_number}"

    for root, dirs, files in os.walk(output_folder):
        for dir in dirs:
            if re.search(search_pattern, dir):
                dir_path = os.path.join(root, dir)
                for sub_root, sub_dirs, sub_files in os.walk(dir_path):
                    for file in sub_files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            image_files.append(os.path.join(sub_root, file))

    if image_files:
        console.print(f"[bold green]Found {len(image_files)} images for ordinance {ordinance_number}:[/bold green]")
        for image_path in image_files:
            console.print(f"  - {image_path}")
        
        # Create a directory to copy the images
        image_output_dir = os.path.join(output_folder, f"Retrieved_Images_Ordinance_{ordinance_number}")
        os.makedirs(image_output_dir, exist_ok=True)

        # Copy images to the new directory
        for image_path in image_files:
            shutil.copy2(image_path, image_output_dir)
        
        console.print(f"[bold green]Images copied to: {image_output_dir}[/bold green]")
    else:
        console.print(f"[bold red]No images found for ordinance number {ordinance_number}.[/bold red]")

    logging.info(f"Retrieved images for ordinance {ordinance_number}")
    return image_files

# Add these API routes:
@app.post("/api/create_digest", response_model=List[dict])
async def api_create_digest(request: DigestRequest):
    try:
        digests = create_digest(request.start_doc_num, request.end_doc_num, request.max_tokens)
        # For each digest, read the content of the file and include it in the response
        for digest in digests:
            file_path = os.path.join(config["Settings"]["DEFAULT_OUTPUT_FOLDER"], digest['filename'])
            with open(file_path, 'r', encoding='utf-8') as f:
                digest['content'] = f.read()
        return digests
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve_document_text", response_model=str)
async def api_retrieve_document_text(request: DocumentTextRequest):
    try:
        document_text = retrieve_document_text(request.document_number)
        if document_text is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return document_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve_images", response_model=List[ImageResponse])
async def api_retrieve_images(request: ImageRetrievalRequest):
    try:
        image_files = retrieve_images(request.ordinance_number)
        image_responses = []
        for image_path in image_files:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_responses.append(ImageResponse(
                    filename=os.path.basename(image_path),
                    content=encoded_image
                ))
        return image_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update the display_menu function:
def display_menu():
    table = Table(title="[bold blue]CREWBRAIN AI-Forensics File Processor[/bold blue]")
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_row("1", "[bold magenta]Process Directory:[/bold magenta] This option processes a directory of files and extracts the text from the sub-folders")
    table.add_row("2", "[bold magenta]Sort Output Files:[/bold magenta] This option sorts the files into the output folder by ordinal number")
    table.add_row("3", "[bold magenta]Create Digest:[/bold magenta] This option creates a digest of the OCR results")
    table.add_row("4", "[bold magenta]Retrieve Document Text:[/bold magenta] This option retrieves the text from a specific document")
    table.add_row("5", "[bold magenta]Retrieve Images:[/bold magenta] This option retrieves images for a specific ordinance")
    table.add_row("6", "[bold magenta]Start API Server:[/bold magenta] This option starts the FastAPI server for API access")
    table.add_row("7", "[bold magenta]Exit[/bold magenta]")
    print(table)

# Update the main function:
def main():
    console = Console()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        display_menu()

        choice = Prompt.ask("[bold green]Enter your choice[/bold green]", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            process_directory()
        elif choice == "2":
            sort_output_files()
        elif choice == "3":
            create_digest()
        elif choice == "4":
            retrieve_document_text()
        elif choice == "5":
            retrieve_images()
        elif choice == "6":
            console.print("[bold green]Starting API server...[/bold green]")
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        elif choice == "7":
            console.print("[bold green]Exiting AI-Forensics File Processor.[/bold green]")
            break
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()