"""Metadata verification: flags signs of tampering or AI generation.

Returns a suspicion score in [0, 1] (higher = more suspicious) plus
human-readable flags, so the Feature Fusion Engine can use it as one feature.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import ExifTags, Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
EDIT_TOOLS = ("photoshop", "gimp", "adobe", "canva", "lavf", "ffmpeg")
AI_HINTS = ("stable diffusion", "midjourney", "dall-e", "firefly",
            "comfyui", "automatic1111", "parameters", "prompt",
            "trainedalgorithmicmedia")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_meta(path):
    img = Image.open(path)
    exif = img.getexif()
    meta = {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in exif.items()}
    try:
        sub = exif.get_ifd(0x8769)  # Exif sub-IFD: DateTimeOriginal etc.
        meta.update({ExifTags.TAGS.get(k, str(k)): str(v) for k, v in sub.items()})
    except Exception:
        pass
    # PNG text chunks / other info (where SD tools store prompts)
    meta.update({str(k): str(v) for k, v in img.info.items()
                 if isinstance(v, (str, int, float))})
    return {k.lower(): v for k, v in meta.items()}


def _media_meta(path):
    if not shutil.which("ffprobe"):
        return {}
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(path)], capture_output=True, text=True)
    try:
        tags = json.loads(out.stdout).get("format", {}).get("tags", {})
    except json.JSONDecodeError:
        return {}
    return {k.lower(): str(v) for k, v in tags.items()}


def check_metadata(path):
    """Return {'metadata_score': 0..1, 'flags': [...], 'sha256': ..., ...}."""
    path = Path(path)
    is_image = path.suffix.lower() in IMAGE_EXT
    try:
        meta = _image_meta(path) if is_image else _media_meta(path)
    except Exception as exc:  # corrupt / unsupported file
        return {"file": path.name, "sha256": sha256_of(path),
                "metadata_score": 0.5,
                "flags": [f"Could not parse metadata: {exc}"], "metadata": {}}

    score, flags = 0.0, []
    text = " ".join(f"{k} {v}" for k, v in meta.items()).lower()

    if not meta:
        score += 0.3
        flags.append("No metadata found (stripped or generated)")
    if any(t in text for t in EDIT_TOOLS):
        score += 0.3
        flags.append("Editing/re-encoding software tag present")
    if any(t in text for t in AI_HINTS):
        score += 0.6
        flags.append("AI-generation hint found in metadata")
    if is_image and meta:
        if "make" not in meta and "model" not in meta:
            score += 0.1
            flags.append("No camera make/model")
        if ("datetime" in meta and "datetimeoriginal" in meta
                and meta["datetime"] != meta["datetimeoriginal"]):
            score += 0.2
            flags.append("Modified time differs from capture time")

    return {
        "file": path.name,
        "sha256": sha256_of(path),
        "metadata_score": round(min(score, 1.0), 2),
        "flags": flags,
        "metadata": dict(list(meta.items())[:20]),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(check_metadata(sys.argv[1]), indent=2))
