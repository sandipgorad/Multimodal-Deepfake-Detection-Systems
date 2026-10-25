"""Reverse image search (local): perceptual-hash index of known images.

Finds near-duplicates (resized / recompressed / lightly edited copies) of
images you have indexed, e.g. original photos or known-real sources, so a
suspicious upload can be matched to its original. Pluggable: add a function
that calls an external API (Google Lens, TinEye) next to `search`.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size))
    px = np.asarray(img, dtype=np.int16)
    return "".join("1" if b else "0" for b in (px[:, 1:] > px[:, :-1]).flatten())


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


class ImageIndex:
    def __init__(self, path="data/image_index.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else {}

    def add(self, image_path, label="known"):
        self.items[str(image_path)] = {"hash": dhash(image_path), "label": label}
        self.path.write_text(json.dumps(self.items))

    def search(self, image_path, max_distance=10):
        q = dhash(image_path)
        hits = [{"path": p, "label": v["label"], "distance": hamming(q, v["hash"])}
                for p, v in self.items.items()]
        return sorted([h for h in hits if h["distance"] <= max_distance],
                      key=lambda h: h["distance"])
