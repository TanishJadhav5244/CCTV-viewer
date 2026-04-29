"""
attributes.py — Zero-shot individual attribute detection using CLIP.

For each detected person crop we classify:
  • Upper/lower clothing colour
  • Apparent gender
  • Accessories (bag, hat, glasses)
  • Age group

All classification is done with CLIP's zero-shot capability — no extra
model weights required beyond what is already loaded in embedder.py.
"""

import numpy as np
import torch
from typing import Dict
from PIL import Image

# ── Candidate label sets ─────────────────────────────────────────────────────

COLOURS = [
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown", "black", "white", "grey", "navy",
    "beige", "cyan", "maroon",
]

UPPER_PROMPTS = [f"a person wearing a {c} top" for c in COLOURS]
LOWER_PROMPTS = [f"a person wearing {c} pants or skirt" for c in COLOURS]

GENDER_PROMPTS = [
    "a photo of a man",
    "a photo of a woman",
]

AGE_PROMPTS = [
    "a photo of a child",
    "a photo of a young adult",
    "a photo of a middle-aged adult",
    "a photo of an elderly person",
]
AGE_LABELS = ["child", "young", "adult", "elderly"]

ACCESSORY_PROMPTS = {
    "has_bag":     ("a person carrying a bag or backpack",
                    "a person with no bag"),
    "has_hat":     ("a person wearing a hat or cap",
                    "a person with no hat"),
    "has_glasses": ("a person wearing glasses or sunglasses",
                    "a person with no glasses"),
}


# ── Extractor ────────────────────────────────────────────────────────────────

class AttributeExtractor:
    """
    Uses the already-loaded CLIP model to classify attributes.
    Only active on 'person' detections to keep processing fast.
    """

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._device = None

        # Pre-encoded text vectors (lazy, built on first call)
        self._upper_vecs = None
        self._lower_vecs = None
        self._gender_vecs = None
        self._age_vecs = None
        self._acc_vecs: Dict[str, np.ndarray] = {}

    def _ensure_loaded(self):
        """Borrow the already-loaded CLIP model from embedder."""
        if self._model is not None:
            return
        from embedder import embedder
        embedder.load()
        self._model = embedder._model
        self._preprocess = embedder._preprocess
        self._tokenize = embedder._tokenize
        self._device = embedder._device
        self._build_text_vecs()

    def _encode_texts(self, prompts):
        with torch.no_grad():
            tokens = self._tokenize(prompts).to(self._device)
            vecs = self._model.encode_text(tokens)
            vecs /= vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy()

    def _build_text_vecs(self):
        self._upper_vecs  = self._encode_texts(UPPER_PROMPTS)
        self._lower_vecs  = self._encode_texts(LOWER_PROMPTS)
        self._gender_vecs = self._encode_texts(GENDER_PROMPTS)
        self._age_vecs    = self._encode_texts(AGE_PROMPTS)
        for key, (pos, neg) in ACCESSORY_PROMPTS.items():
            self._acc_vecs[key] = self._encode_texts([pos, neg])

    def _encode_image(self, pil_img: Image.Image) -> np.ndarray:
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            vec = self._model.encode_image(tensor)
            vec /= vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy()

    def _top1(self, img_vec: np.ndarray, text_vecs: np.ndarray,
               labels: list) -> str:
        sims = (img_vec @ text_vecs.T).flatten()
        return labels[int(np.argmax(sims))]

    def _binary(self, img_vec: np.ndarray, vecs: np.ndarray,
                 threshold: float = 0.0) -> bool:
        sims = (img_vec @ vecs.T).flatten()
        # positive class wins when its score is strictly higher
        return bool(sims[0] > sims[1])

    def extract(self, pil_img: Image.Image, label: str) -> Dict:
        """
        Returns an attribute dict for a single crop.
        Non-person labels get a minimal dict so DB schema stays uniform.
        """
        if label != "person":
            return {"label": label}

        self._ensure_loaded()
        img_vec = self._encode_image(pil_img)

        gender = self._top1(img_vec, self._gender_vecs, ["male", "female"])
        age    = self._top1(img_vec, self._age_vecs,    AGE_LABELS)
        upper  = self._top1(img_vec, self._upper_vecs,  COLOURS)
        lower  = self._top1(img_vec, self._lower_vecs,  COLOURS)

        acc = {k: self._binary(img_vec, v)
               for k, v in self._acc_vecs.items()}

        return {
            "gender":      gender,
            "age_group":   age,
            "upper_color": upper,
            "lower_color": lower,
            **acc,
        }


# Singleton
attribute_extractor = AttributeExtractor()
