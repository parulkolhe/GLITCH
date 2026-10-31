"""Validate and normalize browser PNG drawings to the Quick, Draw! bitmap format."""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError


MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
MAX_IMAGE_PIXELS = 4_000_000
MODEL_IMAGE_SIZE = 28


class DrawingInputError(ValueError):
    """Raised when a request does not contain a usable drawing PNG."""


def _decode_png(image_base64: str) -> Image.Image:
    encoded = image_base64.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or "image/png" not in header.lower():
            raise DrawingInputError("Only PNG data URLs are accepted.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DrawingInputError("Image payload is not valid base64.") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise DrawingInputError("PNG payload is empty or exceeds the 2 MB limit.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                raise DrawingInputError("Image payload must be a PNG.")
            width, height = source.size
            if width < 1 or height < 1 or width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise DrawingInputError("PNG dimensions must be between 1 and 2048 pixels per side.")
            if width * height > MAX_IMAGE_PIXELS:
                raise DrawingInputError("PNG contains too many pixels.")
            source.load()
            if "A" in source.getbands():
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                source = Image.alpha_composite(white, rgba)
            return source.convert("L")
    except DrawingInputError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise DrawingInputError("Could not decode a valid PNG image.") from exc


INK_THRESHOLD = 245          # grayscale value below which a pixel counts as a mark
WORK_SIZE = 256              # intermediate square resolution for stroke normalisation
# Tuned by ml/tune_preprocessing.py on simulated canvas drawings from the VALIDATION
# split (see reports/preprocessing_tuning.json); the held-out test split was scored once.
MARGIN = 0.12                # padding around the drawing, as a fraction of its larger side
STROKE_DILATION = 9          # odd MaxFilter kernel at WORK_SIZE; thickens thin pen strokes


def ink_to_bitmap(ink: np.ndarray, margin: float = None, dilation: int = None) -> np.ndarray:
    """Binary ink mask (any size) -> 28x28 uint8 bitmap in Quick, Draw! format.

    1. Crop to the drawing's bounding box and pad it to a centred square.
    2. Resize to WORK_SIZE and dilate, so thin pen strokes survive downsampling.
    3. Area-resize to 28x28 and rescale so the strongest stroke pixel is 255
       (white ink on black, the dataset's polarity and contrast).
    """
    margin = MARGIN if margin is None else margin
    dilation = STROKE_DILATION if dilation is None else dilation
    if not ink.any():
        raise DrawingInputError("The drawing is blank; add a mark before submitting.")
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    pad = max(2, int(round(max(width, height) * margin)))
    side = max(width, height) + 2 * pad
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = crop.astype(np.uint8) * 255
    work = Image.fromarray(square, mode="L").resize((WORK_SIZE, WORK_SIZE), Image.Resampling.BOX)
    work = work.point(lambda value: 255 if value > 0 else 0)
    if dilation > 1:
        work = work.filter(ImageFilter.MaxFilter(dilation))
    small = np.asarray(work.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.BOX), dtype=np.float32)
    peak = float(small.max())
    if peak <= 0:
        raise DrawingInputError("The drawing is too faint to classify.")
    bitmap = np.clip(np.rint(small * (255.0 / peak)), 0, 255).astype(np.uint8)
    return bitmap.reshape(1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 1)


def preprocess_png(image_base64: str) -> np.ndarray:
    """Convert a canvas PNG into a 28x28 uint8 bitmap in Quick, Draw! style.

    Binarise first: every non-white pixel is ink, so colour, pencil opacity and
    crayon texture do not change what the model sees. Then ink_to_bitmap().
    """
    gray = np.asarray(_decode_png(image_base64), dtype=np.uint8)
    return ink_to_bitmap(gray < INK_THRESHOLD)
