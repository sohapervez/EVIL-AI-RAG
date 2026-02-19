"""Multimodal PDF parser: extracts text, images (OCR + optional vision), and tables.

Uses PyMuPDF for image/table extraction and pymupdf4llm for improved
page-layout-aware text extraction (respects columns, headers, reading order).
Falls back to plain PyMuPDF text extraction if pymupdf4llm is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

import config

logger = logging.getLogger(__name__)

# Try to import pymupdf4llm for better layout analysis
try:
    import pymupdf4llm

    _HAS_LAYOUT = True
    logger.info("pymupdf4llm available — using improved layout analysis")
except ImportError:
    _HAS_LAYOUT = False
    logger.info("pymupdf4llm not installed — using basic PyMuPDF text extraction")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ParsedElement:
    """A single extracted element from a PDF page."""

    content: str
    content_type: str  # "text" | "image_ocr" | "image_description" | "table"
    source: str  # filename
    page: int  # 1-indexed
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lazy-loaded singletons to avoid heavy imports at module level
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_ocr_reader():
    """Lazy-load EasyOCR reader (heavy init, only done once).

    On Apple Silicon Macs, EasyOCR cannot use MPS directly and the
    ``pin_memory`` flag causes a noisy warning.  We suppress it and
    fall back to CPU which is still fast enough for OCR.
    """
    global _ocr_reader
    if _ocr_reader is None:
        import warnings

        import easyocr

        # Suppress the MPS pin_memory warning on Apple Silicon
        warnings.filterwarnings("ignore", message=".*pin_memory.*")

        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return extracted text."""
    try:
        reader = _get_ocr_reader()
        results = reader.readtext(image_bytes)
        text = " ".join([entry[1] for entry in results]).strip()
        return text
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""


def _describe_image_with_vision(image_bytes: bytes) -> str:
    """Use Ollama LLaVA (or similar) to produce a natural-language description."""
    try:
        import base64

        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="llava",
            base_url=config.OLLAMA_BASE_URL,
        )
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this figure from a research paper in detail. Include any data, labels, axes, or trends visible."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        logger.warning("Vision description failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def parse_pdf(pdf_path: str | Path, use_vision: Optional[bool] = None) -> list[ParsedElement]:
    """Parse a single PDF and return a list of ParsedElement objects.

    Extracts:
    - Text blocks per page
    - Images  -> OCR text + optional vision description
    - Tables  -> Markdown representation
    """
    pdf_path = Path(pdf_path)
    if use_vision is None:
        use_vision = config.USE_VISION_DESCRIPTIONS

    filename = pdf_path.name
    elements: list[ParsedElement] = []

    # Try to open PDF with error recovery
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", filename, e)
        # Try opening with repair flag
        try:
            doc = fitz.open(str(pdf_path), filetype="pdf", stream=None)
        except Exception as e2:
            logger.error("Failed to open PDF %s even with repair: %s", filename, e2)
            return elements  # Return empty list

    # --- Layout-aware text extraction (whole document, per page) ---
    layout_texts: dict[int, str] = {}  # page_num -> markdown text
    if _HAS_LAYOUT:
        try:
            md_pages = pymupdf4llm.to_markdown(str(pdf_path), pages=None, page_chunks=True)
            for page_data in md_pages:
                pnum = page_data.get("metadata", {}).get("page", 0)
                # pymupdf4llm page numbers are 0-indexed
                layout_texts[pnum + 1] = page_data.get("text", "")
        except Exception as e:
            # Only log as debug - pymupdf4llm can fail on some PDFs, fallback is fine
            logger.debug("pymupdf4llm extraction failed for %s, falling back to basic: %s", filename, e)

    # Extract text page by page with error handling
    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-indexed
            try:
                page = doc[page_idx]
            except Exception as e:
                logger.warning("Failed to access page %d of %s: %s", page_num, filename, e)
                continue

            # --- Text extraction ---
            try:
                # Prefer pymupdf4llm layout-aware markdown if available
                if page_num in layout_texts and layout_texts[page_num].strip():
                    text = layout_texts[page_num].strip()
                else:
                    text = page.get_text("text")
            except Exception as e:
                logger.warning("Failed to extract text from page %d of %s: %s", page_num, filename, e)
                # Try alternative extraction method
                try:
                    text = page.get_text("dict")
                    # Extract text from text blocks
                    text_blocks = []
                    for block in text.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    text_blocks.append(span.get("text", ""))
                    text = " ".join(text_blocks).strip()
                except Exception as e2:
                    logger.warning("Alternative extraction also failed for page %d of %s: %s", page_num, filename, e2)
                    continue

            if text.strip():
                elements.append(
                    ParsedElement(
                        content=text.strip(),
                        content_type="text",
                        source=filename,
                        page=page_num,
                    )
                )

            # --- Image extraction (skip if disabled for faster processing) ---
            if config.EXTRACT_IMAGES:
                images = page.get_images(full=True)
            else:
                images = []  # Skip image extraction for faster processing

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)

                    # Skip tiny images (likely icons/decorations)
                    if pix.width < 50 or pix.height < 50:
                        continue

                    # Convert any non-RGB/Gray colorspace (CMYK, DeviceN, etc.) to RGB
                    # PNG only supports Grayscale and RGB
                    if pix.colorspace and pix.colorspace.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Strip alpha channel if present (PNG tobytes can fail otherwise)
                    if pix.alpha:
                        pix = fitz.Pixmap(pix, 0)  # 0 = drop alpha

                    image_bytes = pix.tobytes("png")

                    # OCR pass (skip if disabled for faster processing)
                    ocr_text = ""
                    if config.USE_IMAGE_OCR:
                        ocr_text = _ocr_image_bytes(image_bytes)
                    if ocr_text:
                        elements.append(
                            ParsedElement(
                                content=ocr_text,
                                content_type="image_ocr",
                                source=filename,
                                page=page_num,
                                metadata={"image_index": img_idx},
                            )
                        )

                    # Vision description (optional)
                    if use_vision:
                        description = _describe_image_with_vision(image_bytes)
                        if description:
                            elements.append(
                                ParsedElement(
                                    content=description,
                                    content_type="image_description",
                                    source=filename,
                                    page=page_num,
                                    metadata={"image_index": img_idx},
                                )
                            )
                except Exception as e:
                    logger.warning("Image extraction failed on page %d, img %d: %s", page_num, img_idx, e)

            # --- Table extraction (skip if disabled for faster processing) ---
            if config.EXTRACT_TABLES:
                try:
                    tables = page.find_tables()
                    # Handle case where find_tables() returns None or empty list
                    if tables is None:
                        tables = []
                    elif not isinstance(tables, (list, tuple)):
                        # If it's not iterable, skip table extraction for this page
                        tables = []

                    for tbl_idx, table in enumerate(tables):
                        try:
                            df = table.to_pandas()
                            # Convert to markdown string
                            md = df.to_markdown(index=False)
                            if md and md.strip():
                                elements.append(
                                    ParsedElement(
                                        content=f"Table (page {page_num}):\n{md}",
                                        content_type="table",
                                        source=filename,
                                        page=page_num,
                                        metadata={"table_index": tbl_idx},
                                    )
                                )
                        except Exception as e:
                            # Skip this specific table but continue with others
                            logger.debug("Failed to extract table %d on page %d: %s", tbl_idx, page_num, e)
                            continue
                except Exception as e:
                    # Only log as debug to reduce noise (many PDFs don't have extractable tables)
                    logger.debug("Table extraction failed on page %d: %s", page_num, e)
    finally:
        doc.close()

    logger.info("Parsed %s: %d elements extracted", filename, len(elements))
    return elements
