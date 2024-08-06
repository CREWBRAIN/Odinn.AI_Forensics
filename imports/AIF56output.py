# AI Forensics - AIF56
# Output Management

import os
import json
from PIL import Image

def get_subfolder_path(out_folder, fname):
    base_name = os.path.basename(fname).rsplit('.', 1)[0]
    return os.path.join(out_folder, base_name)

def save_markdown(out_folder, base_name, page_num, full_text, images, out_metadata, page_image_with_bboxes):
    subfolder_path = get_subfolder_path(out_folder, base_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Create cleaned_text and image subfolders
    cleaned_text_folder = os.path.join(subfolder_path, "cleaned_text")
    images_folder = os.path.join(subfolder_path, "image")
    os.makedirs(cleaned_text_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # Save text files
    text_filename = os.path.join(cleaned_text_folder, f"{base_name}_pg{page_num}.txt")
    with open(text_filename, "w+", encoding='utf-8') as f:
        f.write(full_text)

    # Save metadata
    out_meta_filepath = text_filename.rsplit(".", 1)[0] + "_meta.json"
    with open(out_meta_filepath, "w+") as f:
        f.write(json.dumps(out_metadata, indent=4))

    # Save images
    for filename, image in images.items():
        image_filename = os.path.join(images_folder, filename)
        image.save(image_filename, "JPEG")

    # Save the page image with bounding boxes
    for filename, image in page_image_with_bboxes.items():
        image_filename = os.path.join(images_folder, filename)
        image.save(image_filename, "JPEG")

    return subfolder_path