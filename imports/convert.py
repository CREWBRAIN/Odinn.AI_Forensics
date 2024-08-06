import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Filter torch pytree user warnings

import pypdfium2 as pdfium # Needs to be at the top to avoid warnings
from PIL import Image, ImageDraw

from marker.utils import flush_cuda_memory
from marker.tables.table import format_tables
from marker.debug.data import dump_bbox_debug_data
from marker.layout.layout import surya_layout
from imports.layout import annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.ocr.lang import replace_langs_with_codes, validate_langs
from marker.ocr.detection import surya_detection
from marker.ocr.recognition import run_ocr
from imports.extract_text import get_text_blocks
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.equations.equations import replace_equations
from marker.pdf.utils import find_filetype
from marker.postprocessors.editor import edit_full_text
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks
from marker.cleaners.fontstyle import find_bold_italic
from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
from marker.cleaners.text import cleanup_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict

from typing import List, Dict, Tuple, Optional
from marker.settings import settings

import pdf2image
from rich.progress import Progress


def setup_language(langs: Optional[List[str]], metadata: Optional[Dict]) -> List[str]:
    """
    Sets up the language configuration for OCR processing.

    This function takes an optional list of languages and metadata dictionary. If the languages list is not provided,
    it defaults to the default language specified in the settings. If metadata is provided, it attempts to extract
    the languages from the metadata. The function then replaces the language names with their corresponding codes
    and validates the final list of languages.

    Args:
        langs (Optional[List[str]]): A list of language names or codes. If None, defaults to the default language.
        metadata (Optional[Dict]): A dictionary containing metadata, which may include language information.

    Returns:
        List[str]: A list of validated language codes.
    """
    if langs is None:
        langs = [settings.DEFAULT_LANG]

    if metadata:
        langs = metadata.get("languages", langs)

    langs = replace_langs_with_codes(langs)
    validate_langs(langs)
    return langs


def detect_file_type(fname: str) -> str:
    """
    Detects the file type of the given file.

    This function takes the file name as input and determines its file type using the find_filetype function.

    Args:
        fname (str): The name of the file whose type needs to be detected.

    Returns:
        str: The detected file type as a string.
    """
    return find_filetype(fname)


def extract_text_blocks(doc, fname: str, max_pages: int, start_page: int) -> Tuple[List, List]:
    """
    Extracts text blocks from the given PDF document.

    This function takes a PDF document, file name, maximum number of pages, and start page as input. It extracts
    text blocks from the document using the get_text_blocks function.

    Args:
        doc: The PDF document object.
        fname (str): The name of the PDF file.
        max_pages (int): The maximum number of pages to process.
        start_page (int): The page number to start processing from.

    Returns:
        Tuple[List, List]: A tuple containing a list of pages and a list of table of contents (TOC) entries.
    """
    return get_text_blocks(doc, fname, max_pages=max_pages, start_page=start_page)


def unpack_models(model_lst: List):
    """
    Unpacks the list of models.

    This function takes a list of models and returns the unpacked models.

    Args:
        model_lst (List): A list of models to be unpacked.

    Returns:
        The unpacked models.
    """
    return model_lst


def identify_text_lines(doc, pages, detection_model, batch_multiplier: int):
    """
    Identifies text lines in the given PDF document.

    This function takes a PDF document, list of pages, detection model, and batch multiplier as input. It identifies
    text lines in the document using the surya_detection function and flushes the CUDA memory.

    Args:
        doc: The PDF document object.
        pages: A list of pages in the document.
        detection_model: The model used for text line detection.
        batch_multiplier (int): The batch multiplier for processing.

    Returns:
        None
    """
    surya_detection(doc, pages, detection_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()


def run_ocr_on_pages(doc, pages, langs, ocr_model, batch_multiplier: int) -> Tuple[List, Dict]:
    """
    Runs OCR on the given pages of the PDF document.

    This function takes a PDF document, list of pages, languages, OCR model, and batch multiplier as input. It runs
    OCR on the pages using the run_ocr function, flushes the CUDA memory, and returns the processed pages and OCR
    statistics.

    Args:
        doc: The PDF document object.
        pages: A list of pages in the document.
        langs: A list of languages for OCR.
        ocr_model: The model used for OCR.
        batch_multiplier (int): The batch multiplier for processing.

    Returns:
        Tuple[List, Dict]: A tuple containing the processed pages and OCR statistics.
    """
    pages, ocr_stats = run_ocr(doc, pages, langs, ocr_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()
    return pages, ocr_stats


def analyze_layout(doc, pages, layout_model, batch_multiplier: int):
    """
    Analyzes the layout of the given PDF document.

    This function takes a PDF document, list of pages, layout model, and batch multiplier as input. It analyzes
    the layout of the document using the surya_layout function and flushes the CUDA memory.

    Args:
        doc: The PDF document object.
        pages: A list of pages in the document.
        layout_model: The model used for layout analysis.
        batch_multiplier (int): The batch multiplier for processing.

    Returns:
        None
    """
    surya_layout(doc, pages, layout_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()


def process_code_and_table_blocks(pages) -> Tuple[int, int]:
    """
    Processes code and table blocks in the given pages.

    This function takes a list of pages as input. It identifies code blocks, indents the blocks, formats tables,
    and returns the count of code blocks and tables.

    Args:
        pages: A list of pages in the document.

    Returns:
        Tuple[int, int]: A tuple containing the count of code blocks and tables.
    """
    code_block_count = identify_code_blocks(pages)
    indent_blocks(pages)
    table_count = format_tables(pages)
    return code_block_count, table_count


def post_process_text(doc, pages, texify_model, batch_multiplier: int) -> Tuple[List, Dict]:
    """
    Post-processes the text in the given PDF document.

    This function takes a PDF document, list of pages, texify model, and batch multiplier as input. It replaces
    equations in the document using the replace_equations function, flushes the CUDA memory, and returns the
    filtered pages and equation statistics.

    Args:
        doc: The PDF document object.
        pages: A list of pages in the document.
        texify_model: The model used for text post-processing.
        batch_multiplier (int): The batch multiplier for processing.

    Returns:
        Tuple[List, Dict]: A tuple containing the filtered pages and equation statistics.
    """
    filtered, eq_stats = replace_equations(doc, pages, texify_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()
    return filtered, eq_stats


def extract_images_from_pdf(doc, pages):
    """
    Extracts images from the given PDF document.

    This function takes a PDF document and list of pages as input. If image extraction is enabled in the settings,
    it extracts images from the document using the extract_images function.

    Args:
        doc: The PDF document object.
        pages: A list of pages in the document.

    Returns:
        None
    """
    if settings.EXTRACT_IMAGES:
        extract_images(doc, pages)


def finalize_text(pages, filtered, edit_model, batch_multiplier: int) -> Tuple[str, Dict]:
    """
    Finalizes the text in the given pages.

    This function takes a list of pages, filtered pages, edit model, and batch multiplier as input. It performs
    various text post-processing steps, including splitting heading blocks, finding bold and italic text, merging
    spans and lines, filtering common titles, cleaning up text, replacing bullets, and editing the full text. It
    flushes the CUDA memory and returns the final full text and edit statistics.

    Args:
        pages: A list of pages in the document.
        filtered: A list of filtered pages.
        edit_model: The model used for text editing.
        batch_multiplier (int): The batch multiplier for processing.

    Returns:
        Tuple[str, Dict]: A tuple containing the final full text and edit statistics.
    """
    split_heading_blocks(pages)
    find_bold_italic(pages)

    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    full_text = cleanup_text(full_text)
    full_text = replace_bullets(full_text)

    full_text, edit_stats = edit_full_text(full_text, edit_model, batch_multiplier=batch_multiplier)
    flush_cuda_memory()
    return full_text, edit_stats


def convert_single_pdf(
        fname: str,
        model_lst: List,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    """
    Converts a single PDF file to text and images with metadata.

    This function takes a file name, list of models, maximum number of pages, start page, metadata, languages, and
    batch multiplier as input. It processes the PDF file in multiple steps, including language setup, file type
    detection, text block extraction, model unpacking, text line identification, OCR, layout analysis, code and
    table block processing, text post-processing, and image extraction. It returns the final full text, document
    images, and metadata.

    Args:
        fname (str): The name of the PDF file to be converted.
        model_lst (List): A list of models to be used for processing.
        max_pages (int, optional): The maximum number of pages to process. Defaults to None.
        start_page (int, optional): The page number to start processing from. Defaults to None.
        metadata (Optional[Dict], optional): A dictionary containing metadata. Defaults to None.
        langs (Optional[List[str]], optional): A list of languages for OCR. Defaults to None.
        batch_multiplier (int, optional): The batch multiplier for processing. Defaults to 1.

    Returns:
        Tuple[str, Dict[str, Image.Image], Dict]: A tuple containing the final full text, document images, and metadata.
    """
    with Progress() as progress:
        task = progress.add_task("[green]Processing PDF...", total=11)

        # 1. Language Setup
        langs = setup_language(langs, metadata)
        progress.update(task, advance=1)

        # 2. File Type Detection
        filetype = detect_file_type(fname)
        out_meta = {"languages": langs, "filetype": filetype}
        if filetype == "other":
            return "", {}, out_meta
        progress.update(task, advance=1)

        # 3. Text Block Extraction
        doc = pdfium.PdfDocument(fname)
        pages, toc = extract_text_blocks(doc, fname, max_pages, start_page)
        out_meta.update({"toc": toc, "pages": len(pages)})
        if start_page:
            for page_idx in range(start_page):
                doc.del_page(0)
        progress.update(task, advance=1)

        # 4. Model Unpacking
        texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = unpack_models(model_lst)
        progress.update(task, advance=1)

        # 5. Text Line Identification
        identify_text_lines(doc, pages, detection_model, batch_multiplier)
        pages, ocr_stats = run_ocr_on_pages(doc, pages, langs, ocr_model, batch_multiplier)
        out_meta["ocr_stats"] = ocr_stats
        if len([b for p in pages for b in p.blocks]) == 0:
            print(f"Could not extract any text blocks for {fname}")
            return "", {}, out_meta
        analyze_layout(doc, pages, layout_model, batch_multiplier)
        progress.update(task, advance=1)

        # 6. Layout Analysis
        bad_span_ids = filter_header_footer(pages)
        out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}
        annotate_block_types(pages)
        dump_bbox_debug_data(doc, fname, pages)
        surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
        sort_blocks_in_reading_order(pages)
        flush_cuda_memory()
        progress.update(task, advance=1)

        # 7. Code and Table Blocks
        code_block_count, table_count = process_code_and_table_blocks(pages)
        out_meta["block_stats"].update({"code": code_block_count, "table": table_count})
        for page in pages:
            for block in page.blocks:
                block.filter_spans(bad_span_ids)
                block.filter_bad_span_types()
        filtered, eq_stats = post_process_text(doc, pages, texify_model, batch_multiplier)
        out_meta["block_stats"]["equations"] = eq_stats
        progress.update(task, advance=1)

        # 9. Image Extraction
        extract_images_from_pdf(doc, pages)
        progress.update(task, advance=1)

        # 10. Text Post-Processing
        full_text, edit_stats = finalize_text(pages, filtered, edit_model, batch_multiplier)
        out_meta["postprocess_stats"] = {"edit": edit_stats}
        doc_images = images_to_dict(pages)
        progress.update(task, advance=1)

        # 11. Return Values
        return full_text, doc_images, out_meta


def extract_page_images_with_bboxes(
        fname: str,
        model_lst: List,
        max_pages: int = 1,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1
) -> Tuple[Dict[str, Image.Image], Dict]:
    """
    Extracts page images with bounding boxes from a PDF file.

    This function takes a file name, list of models, maximum number of pages, start page, metadata, languages, and
    batch multiplier as input. It processes the PDF file in multiple steps, including language setup, file type
    detection, text block extraction, model unpacking, text line identification, OCR, layout analysis, code and
    table block processing, text post-processing, and image extraction. It returns the annotated page images and
    metadata.

    Args:
        fname (str): The name of the PDF file to be processed.
        model_lst (List): A list of models to be used for processing.
        max_pages (int, optional): The maximum number of pages to process. Defaults to 1.
        start_page (int, optional): The page number to start processing from. Defaults to None.
        metadata (Optional[Dict], optional): A dictionary containing metadata. Defaults to None.
        langs (Optional[List[str]], optional): A list of languages for OCR. Defaults to None.
        batch_multiplier (int, optional): The batch multiplier for processing. Defaults to 1.

    Returns:
        Tuple[Dict[str, Image.Image], Dict]: A tuple containing the annotated page images and metadata.
    """
    with Progress() as progress:
        task = progress.add_task("[green]Extracting page images with bounding boxes...", total=9)

        # 1. Language Setup
        langs = setup_language(langs, metadata)
        progress.update(task, advance=1)

        # 2. File Type Detection
        filetype = detect_file_type(fname)
        out_meta = {"languages": langs, "filetype": filetype}
        if filetype == "other":
            return {}, out_meta
        progress.update(task, advance=1)

        # 3. Text Block Extraction
        doc = pdfium.PdfDocument(fname)
        pages, toc = extract_text_blocks(doc, fname, max_pages, start_page)
        out_meta.update({"toc": toc, "pages": len(pages)})
        if start_page:
            for page_idx in range(start_page):
                doc.del_page(0)
        progress.update(task, advance=1)

        # 4. Model Unpacking
        texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = unpack_models(model_lst)
        progress.update(task, advance=1)

        # 5. Text Line Identification
        identify_text_lines(doc, pages, detection_model, batch_multiplier)
        pages, ocr_stats = run_ocr_on_pages(doc, pages, langs, ocr_model, batch_multiplier)
        out_meta["ocr_stats"] = ocr_stats
        if len([b for p in pages for b in p.blocks]) == 0:
            print(f"Could not extract any text blocks for {fname}")
            return {}, out_meta
        annotated_images = surya_layout(doc, pages, layout_model, batch_multiplier=batch_multiplier)
        flush_cuda_memory()
        progress.update(task, advance=1)

        print(f"Extracted {len(annotated_images)} annotated images.")

        # 6. Layout Analysis
        bad_span_ids = filter_header_footer(pages)
        out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}
        annotate_block_types(pages)
        dump_bbox_debug_data(doc, fname, pages)
        surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
        sort_blocks_in_reading_order(pages)
        flush_cuda_memory()
        progress.update(task, advance=1)

        # 7. Code and Table Blocks
        code_block_count, table_count = process_code_and_table_blocks(pages)
        out_meta["block_stats"].update({"code": code_block_count, "table": table_count})
        for page in pages:
            for block in page.blocks:
                block.filter_spans(bad_span_ids)
                block.filter_bad_span_types()
        filtered, eq_stats = post_process_text(doc, pages, texify_model, batch_multiplier)
        out_meta["block_stats"]["equations"] = eq_stats
        progress.update(task, advance=1)

        # 8. Image Extraction
        extract_images_from_pdf(doc, pages)
        progress.update(task, advance=1)

        # 9. Return Values
        doc_images = images_to_dict(pages)
        return {f"page_{i+1}": img for i, img in enumerate(annotated_images)}, out_meta


def convert_single_pdf_md_only(
        fname: str,
        model_lst: List,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1
) -> Tuple[str, Dict]:
    """
    Converts a single PDF file to Markdown text with metadata.

    This function is designed to process a single PDF file and convert its contents into Markdown text format. 
    It takes several parameters including the file name, a list of models, the maximum number of pages to process, 
    the starting page, metadata, languages, and a batch multiplier. The function follows a series of steps to 
    achieve the conversion, including language setup, file type detection, text block extraction, model unpacking, 
    text line identification, OCR, layout analysis, code and table block processing, and text post-processing. 
    The function returns the final Markdown text and metadata.

    Args:
        fname (str): The name of the PDF file to be converted. This is the primary input to the function and 
                     should be a valid file path to the PDF document that needs to be processed.
        model_lst (List): A list of models to be used for processing. These models are essential for various 
                          stages of the conversion process, including text line identification, OCR, layout 
                          analysis, and text post-processing.
        max_pages (int, optional): The maximum number of pages to process. This parameter allows the user to 
                                   limit the number of pages that will be processed from the PDF document. If 
                                   not provided, the function will process all pages in the document.
        start_page (int, optional): The page number to start processing from. This parameter is useful if the 
                                    user wants to skip a certain number of pages at the beginning of the document. 
                                    If not provided, the function will start processing from the first page.
        metadata (Optional[Dict], optional): A dictionary containing metadata. This metadata can include 
                                             additional information about the document, such as author, title, 
                                             and other relevant details. The function uses this metadata to 
                                             enhance the processing and output.
        langs (Optional[List[str]], optional): A list of languages for OCR. This parameter allows the user to 
                                               specify the languages that should be considered during the OCR 
                                               process. If not provided, the function will use the default 
                                               language settings.
        batch_multiplier (int, optional): The batch multiplier for processing. This parameter controls the 
                                          batch size for various processing steps, allowing the user to adjust 
                                          the performance and resource usage. The default value is 1.

    Returns:
        Tuple[str, Dict]: A tuple containing the final Markdown text and metadata. The Markdown text is the 
                          converted content of the PDF document, and the metadata is a dictionary containing 
                          various details about the processing and the document itself.

    The function follows these steps:

    1. Language Setup:
       - The function first sets up the language configuration for OCR processing. It takes the optional list 
         of languages and metadata dictionary. If the languages list is not provided, it defaults to the default 
         language specified in the settings. If metadata is provided, it attempts to extract the languages from 
         the metadata. The function then replaces the language names with their corresponding codes and validates 
         the final list of languages.

    2. File Type Detection:
       - The function detects the file type of the input PDF document. It updates the metadata with the detected 
         file type and languages. If the file type is not supported (i.e., "other"), the function returns an 
         empty string and the metadata.

    3. Text Block Extraction:
       - The function extracts text blocks from the PDF document. It creates a PdfDocument object and extracts 
         the text blocks from the specified pages. The metadata is updated with the table of contents (TOC) and 
         the number of pages. If a start page is specified, the function deletes the pages before the start page.

    4. Model Unpacking:
       - The function unpacks the models from the provided list. These models are used for various processing 
         steps, including text line identification, OCR, layout analysis, and text post-processing.

    5. Text Line Identification:
       - The function identifies text lines in the document using the detection model. It then runs OCR on the 
         pages using the OCR model and updates the metadata with the OCR statistics. If no text blocks are 
         extracted, the function prints a message and returns an empty string and the metadata.

    6. Layout Analysis:
       - The function performs layout analysis on the document. It filters header and footer spans, annotates 
         block types, dumps bounding box debug data, orders the blocks, and sorts them in reading order. The 
         metadata is updated with the block statistics.

    7. Code and Table Blocks:
       - The function processes code and table blocks in the document. It updates the metadata with the count 
         of code and table blocks. It then filters spans and bad span types in each block. The function performs 
         post-processing on the text and updates the metadata with the equation statistics.

    8. Image Extraction (Skipped):
       - The function skips the image extraction step. This step is commented out in the code.

    9. Text Post-Processing:
       - The function finalizes the text in the document. It performs various text post-processing steps, 
         including splitting heading blocks, finding bold and italic text, merging spans and lines, filtering 
         common titles, cleaning up text, replacing bullets, and editing the full text. The metadata is updated 
         with the post-processing statistics.

    10. Return Values:
        - The function returns the final Markdown text and metadata.

    Example usage:
        >>> fname = "example.pdf"
        >>> model_lst = [texify_model, layout_model, order_model, edit_model, detection_model, ocr_model]
        >>> max_pages = 10
        >>> start_page = 1
        >>> metadata = {"author": "John Doe", "title": "Example PDF"}
        >>> langs = ["en", "fr"]
        >>> batch_multiplier = 2
        >>> markdown_text, metadata = convert_single_pdf_md_only(fname, model_lst, max_pages, start_page, metadata, langs, batch_multiplier)
        >>> print(markdown_text)
        >>> print(metadata)
    """

    with Progress() as progress:
        task = progress.add_task("[green]Processing PDF...", total=11)

        # 1. Language Setup
        langs = setup_language(langs, metadata)
        progress.update(task, advance=1)

        # 2. File Type Detection
        filetype = detect_file_type(fname)
        out_meta = {"languages": langs, "filetype": filetype}
        if filetype == "other":
            return "", out_meta
        progress.update(task, advance=1)

        # 3. Text Block Extraction
        doc = pdfium.PdfDocument(fname)
        pages, toc = extract_text_blocks(doc, fname, max_pages, start_page)
        out_meta.update({"toc": toc, "pages": len(pages)})
        if start_page:
            for page_idx in range(start_page):
                doc.del_page(0)
        progress.update(task, advance=1)

        # 4. Model Unpacking
        texify_model, layout_model, order_model, edit_model, detection_model, ocr_model = unpack_models(model_lst)
        progress.update(task, advance=1)

        # 5. Text Line Identification
        identify_text_lines(doc, pages, detection_model, batch_multiplier)
        pages, ocr_stats = run_ocr_on_pages(doc, pages, langs, ocr_model, batch_multiplier)
        out_meta["ocr_stats"] = ocr_stats
        if len([b for p in pages for b in p.blocks]) == 0:
            print(f"Could not extract any text blocks for {fname}")
            return "", out_meta
        analyze_layout(doc, pages, layout_model, batch_multiplier)
        progress.update(task, advance=1)

        # 6. Layout Analysis
        bad_span_ids = filter_header_footer(pages)
        out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}
        annotate_block_types(pages)
        dump_bbox_debug_data(doc, fname, pages)
        surya_order(doc, pages, order_model, batch_multiplier=batch_multiplier)
        sort_blocks_in_reading_order(pages)
        flush_cuda_memory()
        progress.update(task, advance=1)

        # 7. Code and Table Blocks
        code_block_count, table_count = process_code_and_table_blocks(pages)
        out_meta["block_stats"].update({"code": code_block_count, "table": table_count})
        for page in pages:
            for block in page.blocks:
                block.filter_spans(bad_span_ids)
                block.filter_bad_span_types()
        filtered, eq_stats = post_process_text(doc, pages, texify_model, batch_multiplier)
        out_meta["block_stats"]["equations"] = eq_stats
        progress.update(task, advance=1)

        # 9. Image Extraction (Skipped)
        # extract_images_from_pdf(doc, pages)
        progress.update(task, advance=1)

        # 10. Text Post-Processing
        full_text, edit_stats = finalize_text(pages, filtered, edit_model, batch_multiplier)
        out_meta["postprocess_stats"] = {"edit": edit_stats}
        progress.update(task, advance=1)

        # 11. Return Values
        return full_text, out_meta

# Create a class using pdf2image to extract the page images and then apply the bounding boxes to them.
def pdf_to_images_with_bboxes(fname, model_lst, max_pages=None, start_page=None, langs=None, batch_multiplier=1):
    pass