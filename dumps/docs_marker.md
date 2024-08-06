Marker CLI Usage
First, some configuration:

Inspect the settings in marker/settings.py. You can override any settings with environment variables.
Your torch device will be automatically detected, but you can override this. For example, TORCH_DEVICE=cuda.
If using GPU, set INFERENCE_RAM to your GPU VRAM (per GPU). For example, if you have 16 GB of VRAM, set INFERENCE_RAM=16.
Depending on your document types, marker's average memory usage per task can vary slightly. You can configure VRAM_PER_TASK to adjust this if you notice tasks failing with GPU out of memory errors.
By default, marker will use surya for OCR. Surya is slower on CPU, but more accurate than tesseract. If you want faster OCR, set OCR_ENGINE to ocrmypdf. This also requires external dependencies (see above). If you don't want OCR at all, set OCR_ENGINE to None.
Convert a single file
marker_single /path/to/file.pdf /path/to/output/folder --batch_multiplier 2 --max_pages 10 --langs English
--batch_multiplier is how much to multiply default batch sizes by if you have extra VRAM. Higher numbers will take more VRAM, but process faster. Set to 2 by default. The default batch sizes will take ~3GB of VRAM.
--max_pages is the maximum number of pages to process. Omit this to convert the entire document.
--langs is a comma separated list of the languages in the document, for OCR
Make sure the DEFAULT_LANG setting is set appropriately for your document. The list of supported languages for OCR is here. If you need more languages, you can use any language supported by Tesseract if you set OCR_ENGINE to ocrmypdf. If you don't need OCR, marker can work with any language.

Convert multiple files
marker /path/to/input/folder /path/to/output/folder --workers 10 --max 10 --metadata_file /path/to/metadata.json --min_length 10000
--workers is the number of pdfs to convert at once. This is set to 1 by default, but you can increase it to increase throughput, at the cost of more CPU/GPU usage. Parallelism will not increase beyond INFERENCE_RAM / VRAM_PER_TASK if you're using GPU.
--max is the maximum number of pdfs to convert. Omit this to convert all pdfs in the folder.
--min_length is the minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
--metadata_file is an optional path to a json file with metadata about the pdfs. If you provide it, it will be used to set the language for each pdf. If not, DEFAULT_LANG will be used. The format is:
{
  "pdf1.pdf": {"languages": ["English"]},
  "pdf2.pdf": {"languages": ["Spanish", "Russian"]},
  ...
}
You can use language names or codes. The exact codes depend on the OCR engine. See here for a full list for surya codes, and here for tesseract.

Convert multiple files on multiple GPUs
MIN_LENGTH=10000 METADATA_FILE=../pdf_meta.json NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
METADATA_FILE is an optional path to a json file with metadata about the pdfs. See above for the format.
NUM_DEVICES is the number of GPUs to use. Should be 2 or greater.
NUM_WORKERS is the number of parallel processes to run on each GPU. Per-GPU parallelism will not increase beyond INFERENCE_RAM / VRAM_PER_TASK.
MIN_LENGTH is the minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
Note that the env variables above are specific to this script, and cannot be set in local.env.