"""
OCR — extract text from images/scanned documents.

Uses pytesseract + Pillow. Gracefully handles missing tesseract binary.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.helpers import setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.ocr")


def extract_text_from_image(image_path: str, language: str = "eng") -> dict:
    """OCR on a single image."""
    try:
        from PIL import Image
        import pytesseract

        p = Path(image_path)
        if not p.exists():
            return {"status": "error", "message": f"Image not found: {image_path}"}

        img = Image.open(p)
        text = pytesseract.image_to_string(img, lang=language)

        # Confidence via detailed data
        try:
            data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data["conf"] if int(c) >= 0]
            avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        except Exception:
            avg_conf = None

        word_count = len(text.split())

        return {
            "status": "success",
            "image_path": str(p),
            "extracted_text": text.strip(),
            "confidence": avg_conf,
            "word_count": word_count,
        }

    except ImportError as e:
        return {"status": "error", "message": f"Library not installed: {e}. Install pytesseract and Pillow."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_ocr(folder_path: str, language: str = "eng") -> dict:
    """OCR on all images in a folder."""
    try:
        p = Path(folder_path)
        if not p.is_dir():
            return {"status": "error", "message": f"Not a directory: {folder_path}"}

        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
        image_files = [f for f in p.iterdir() if f.suffix.lower() in extensions]

        if not image_files:
            return {"status": "error", "message": "No image files found in directory"}

        logger.info(f"Batch OCR: {len(image_files)} images in {folder_path}")

        results = []
        for img_file in sorted(image_files):
            r = extract_text_from_image(str(img_file), language)
            results.append(r)

        successful = [r for r in results if r.get("status") == "success"]
        total_words = sum(r.get("word_count", 0) for r in successful)

        return safe_json_serialize({
            "status": "success",
            "total_images": len(image_files),
            "successful": len(successful),
            "failed": len(image_files) - len(successful),
            "total_words_extracted": total_words,
            "results": results,
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def ocr_and_upload(image_path: str, output_name: str = "ocr.json", **kwargs) -> dict:
    """Convenience function: extract_text_from_image + upload."""
    result = extract_text_from_image(image_path, **kwargs)
    upload_result(result, output_name)
    return result
