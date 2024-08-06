#from surya.model.detection import segformer
from imports.model_utils.segformer import load_model as load_segformer_model
from imports.model_utils.segformer import load_processor as load_segformer_processor

#from marker.settings import settings
from imports.model_utils.settings import settings

#from surya.model.recognition.model import load_model as load_recognition_model
from imports.model_utils.recognition_model import load_model as load_recognition_model

#from surya.model.recognition.processor import load_processor as load_recognition_processor
from imports.model_utils.recognition_processor import load_processor as load_recognition_processor


#from texify.model.model import load_model as load_texify_model
from imports.model_utils.textify_model import load_model as load_texify_model


#from texify.model.processor import load_processor as load_texify_processor
from imports.model_utils.textify_model_processor import load_processor as load_texify_processor


#from surya.model.ordering.model import load_model as load_order_model
from imports.model_utils.ordering_model import load_model as load_order_model

#from surya.model.ordering.processor import load_processor as load_order_processor
from imports.model_utils.ordering_processor import load_processor as load_order_processor


#from marker.postprocessors.editor import load_editing_model
from imports.model_utils.marker_editng_model import load_editing_model

import os
import pickle

CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def setup_recognition_model(langs, device=None, dtype=None):
    if device:
        rec_model = load_recognition_model(langs=langs, device=device, dtype=dtype)
    else:
        rec_model = load_recognition_model(langs=langs)
    rec_processor = load_recognition_processor()
    rec_model.processor = rec_processor
    return rec_model


def setup_detection_model(device=None, dtype=None):
    if device:
        model = load_segformer_model(device=device, dtype=dtype)
    else:
        model = load_segformer_model()

    processor = load_segformer_processor()
    model.processor = processor
    return model


def setup_texify_model(device=None, dtype=None):
    if device:
        texify_model = load_texify_model(checkpoint=settings.TEXIFY_MODEL_NAME, device=device, dtype=dtype)
    else:
        texify_model = load_texify_model(checkpoint=settings.TEXIFY_MODEL_NAME, device=settings.TORCH_DEVICE_MODEL, dtype=settings.TEXIFY_DTYPE)
    texify_processor = load_texify_processor()
    texify_model.processor = texify_processor
    return texify_model


def setup_layout_model(device=None, dtype=None):
    if device:
        model = load_segformer_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT, device=device, dtype=dtype)
    else:
        model = load_segformer_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_segformer_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    model.processor = processor
    return model


def setup_order_model(device=None, dtype=None):
    if device:
        model = load_order_model(device=device, dtype=dtype)
    else:
        model = load_order_model()
    processor = load_order_processor()
    model.processor = processor
    return model


def load_all_models(langs=None, device=None, dtype=None, force_load_ocr=False):
    cache_file = os.path.join(CACHE_DIR, "models.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            model_lst = pickle.load(f)
        return model_lst

    if device is not None:
        assert dtype is not None, "Must provide dtype if device is provided"

    detection = setup_detection_model(device, dtype)
    layout = setup_layout_model(device, dtype)
    order = setup_order_model(device, dtype)
    edit = load_editing_model(device, dtype)
    ocr = setup_recognition_model(langs, device, dtype)
    texify = setup_texify_model(device, dtype)
    model_lst = [texify, layout, order, edit, detection, ocr]

    with open(cache_file, "wb") as f:
        pickle.dump(model_lst, f)

    return model_lst

