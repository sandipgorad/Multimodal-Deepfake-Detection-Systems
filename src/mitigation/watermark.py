"""Watermark / content-credential verification.

1. detect_content_credentials: looks for C2PA manifests and IPTC/XMP
   "AI generated" markers that generators (Adobe Firefly, DALL-E, etc.) embed.
2. embed_lsb / extract_lsb: simple invisible watermark for content *you*
   publish (fragile to compression - fine for a demo of the concept).
"""
from pathlib import Path

import numpy as np
from PIL import Image

AI_MARKERS = [b"trainedAlgorithmicMedia", b"compositeSynthetic",
              b"algorithmicMedia", b"Made with AI"]


def detect_content_credentials(path):
    data = Path(path).read_bytes()
    found = {
        "c2pa_manifest": b"c2pa" in data or b"jumb" in data,
        "ai_generated_marker": any(m in data for m in AI_MARKERS),
    }
    found["has_credentials"] = any(found.values())
    return found


def embed_lsb(in_path, out_path, message):
    """Hide `message` in the least-significant bits (saves lossless PNG)."""
    img = np.array(Image.open(in_path).convert("RGB"))
    bits = "".join(f"{b:08b}" for b in message.encode()) + "00000000"
    flat = img.flatten()
    if len(bits) > flat.size:
        raise ValueError("Image too small for this message")
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)
    Image.fromarray(flat.reshape(img.shape)).save(out_path, format="PNG")


def extract_lsb(path, max_chars=256):
    flat = np.array(Image.open(path).convert("RGB")).flatten()
    chars = []
    for c in range(max_chars):
        byte = "".join(str(flat[c * 8 + i] & 1) for i in range(8))
        if byte == "00000000":
            break
        chars.append(int(byte, 2))
    try:
        return bytes(chars).decode()
    except UnicodeDecodeError:
        return ""


def verify_watermark(path, expected):
    return extract_lsb(path) == expected
