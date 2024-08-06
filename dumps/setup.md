# AI Forensics - AIF56 / Installation & TODO

# INSTALLATION
1. Create virtual environment.

2. Install uv: optional pip install uv

3. Install PyTorch: 
    1. Set Extended Timeout: $env:UV_HTTP_TIMEOUT=120 # you have to increase the timeout as the downloads take awhile
    2. Install Torch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        - The command above is for my desktop, you should build your own: PyTorch Build / Stable (2.3.1) / Pip / Python / CUDA 12.1
        - You'll need python 3.9+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine. Build your download command here: https://pytorch.org/get-started/locally
        - Depending on the speed of your connection, regular pip install is more observable for the download of the 2.4gb file it needs.  Example my speed: 0.1/2.4 GB 1.8 MB/s eta 0:21:16
          This way you're not wondering what's happening and can watch the download.
       
       Alternatively, you can just $env:UV_HTTP_TIMEOUT=60, uv pip install torch. # NOTE: note sure this works yet, needs testing

4. Install Marker: uv pip install marker-pdf

5. Install the rest: uv pip install litellm rich openai flask flask_restful python-dotenv pdf2image pytesseract pdf2image

6. Generate a requirements.txt: uv pip freeze > requirements.txt
    - You can generate a `requirements.txt` file by using the `pip freeze` command, which will list all the installed packages in the current environment along with their versions.
    - You can redirect the output to a file by running: pip freeze > requirements.txt
    - This will create a `requirements.txt` file in your current directory with all the necessary information.
    - Remember to activate your project's virtual environment before running this command if you are using one.

7. Install Tesseract on the system and make it's in the env path or windows environment path

8. Ensure your .env and required API keys are in the file.

# TODO:
1. # TODO: Separate Text by Page:
   - When Marker converts, it consolidates all the text into one file. We need each page separate.

2. # TODO: Extract Page Elements:
   - The output should include: Page metadata json, Page text, Page image, Page image with bounding boxes applied, Images found on page, Tables found on page
   # UPDATE: Removing the requirement for Page image with bounding boxes applied. start simple.

3. # TODO: Send Data to Vision Model:
   - Send the following to the vision model for cleaning:
     - Prompt
     - Page text
     - Page image with bounding boxes applied
     - Images found on page
     - Tables found on page
   - Until we figure out the page image with bounding boxes applied, send:
     - Prompt
     - Page text
     - Page image
     - Images found on page
     - Tables found on page
