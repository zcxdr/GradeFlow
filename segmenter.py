# segmenter.py -- PDF rasterisation and image utilities.
# In the current page-by-page pipeline, this module is used for:
#   - pdf_to_images(): convert PDF pages to PIL Images
#   - save_segment(): save a PIL Image to disk as JPEG
#   - extract_roll_number(): parse roll number from PDF filename

import logging
import os
from pathlib import Path
from typing import List

from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 100) -> List[Image.Image]:
    """
    Rasterise every page of a PDF. Returns a list of PIL Images.
    Requires: pip install pdf2image + sudo apt-get install poppler-utils
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required: pip install pdf2image\n"
            "Also install poppler: sudo apt-get install poppler-utils"
        )
    logger.info("Rasterising %s at %d dpi ...", pdf_path, dpi)
    pages = convert_from_path(pdf_path, dpi=dpi, fmt="jpeg")
    logger.info("  -> %d page(s)", len(pages))
    return pages


def extract_roll_number(pdf_filename: str) -> str:
    """
    Extract the roll number from a PDF filename.
    Convention: <ROLL_NUMBER>_<StudentName>.pdf
    e.g. '23UCS600_JATIN JAIN.pdf' -> '23UCS600'
    Override this function if your naming convention differs.
    """
    stem  = Path(pdf_filename).stem
    parts = stem.split("_")
    return parts[0].upper().strip() if parts else stem


def save_segment(
    crop: Image.Image,
    segments_dir: str,
    roll_number: str,
    question_id: str,
    page_number: int,
    suffix: str = "",
) -> str:
    """
    Save a PIL Image to disk as JPEG.
    Returns the absolute file path.
    """
    out_dir = Path(segments_dir) / roll_number / question_id
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "page_{:03d}{}.jpg".format(page_number, suffix)
    filepath = out_dir / filename
    crop.save(str(filepath), format="JPEG", quality=92)
    return str(filepath)
