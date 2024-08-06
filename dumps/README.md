# Rich TUI Structure:

1. **Main Menu:**
   - **Load PDF:**  Prompt the user to select a PDF file.
   - **Settings:**  Open a submenu to configure settings.
   - **Process PDF (Step-by-Step):**  Open a submenu to guide the user through each processing step individually.
   - **Process PDF (Uber Step):**  Run all processing steps automatically and save the state after each step.
   - **Exit:** Quit the application.

2. **Settings Submenu:**
   - **Marker Settings:**
     - Torch device (CPU, GPU, MPS)
     - OCR engine (Surya, OCRMyPDF, None)
     - Languages for OCR
     - Batch multiplier
     - **Bounding Box Options:**
       - Extract bounding boxes (True/False)
       - Bounding box types to extract (list, selectable from available types)
       - Visualize bounding boxes (True/False)
     - Other relevant settings from `marker.settings`
   - **Vision Cleaning Settings:**
     - Model to use (e.g., "ollama/llava-phi3")
     - Prompt structure and metadata inclusion
   - **Output Settings:**
     - Output folder path
     - Citation information (title, author, etc.)
   - Save settings to a file.
   - Load settings from a file.

3. **Step-by-Step Processing Submenu:**
   - **Extract Text and Metadata (Marker):** Run `marker.convert.convert_single_pdf` and save results.
   - **Extract Images:** 
     - Extract page images without bounding boxes.
     - **Extract Images with Bounding Boxes:** Extract page images with selected bounding boxes visualized.
   - **Vision Cleaning:** Process selected pages with vision cleaning.
   - **Generate Output Package:** Create the final output folder structure.

**Additional Notes:**

- When presenting the "Bounding box types to extract" option, use `rich.prompt.Prompt.ask` with `choices` parameter to allow the user to select from a list of available bounding box types from the `marker.schema.bbox` module.
- Ensure that the `extract_image_with_bboxes` function (mentioned in the previous response) is called when the user selects the "Extract Images with Bounding Boxes" option.

--

# AIF
1. Codebase Dump -> Paste AIF ->
2. Task: Convert AI Forensics into a standalone script so it doesnt rely on marker. I want finer control over the settings. we should be able to maximize the marker package. Streamline the user experience. The final set of artifacts should include a big package of the PDF individual page images, individual page images with bounding boxes, individual pages with extracted markdown/text and all neatly packaged up in a folder with sub-folders along with the original PDF. It should be citation and research worthy. What else could we maximize with marker? We should have also user input metadata as well as another option to create metadata from the visual cleaning process. We want to give as much data to that process as possible to improve accuracy. We will use a rich tui to drive the process with menu options for each step, settings for all steps and a way to save the state of the process to a file in case of failure or interruption. Have menu options for each step and one uber step which goes from start to finish with all steps and saves the state of each step to a file. Solidify the core feature set. 
3. Test AIF
4. Create Dify Tool locally first

# Doc_Retriever
1. Test Doc_Retriever
2. Create Dify Tool locally first
3. Test Dify Tool

# Sources:
- Reference folder: D:\Coding\Builds\AIF5\reference


--