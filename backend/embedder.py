import numpy as np
import torch
import open_clip
from PIL import Image
from pathlib import Path
from config import CLIP_MODEL, CLIP_PRETRAINED


class CLIPEmbedder:
    """Singleton CLIP embedder loaded once at startup."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print(f"[CLIP] Loading model {CLIP_MODEL} ({CLIP_PRETRAINED})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model.eval()
        print(f"[CLIP] Ready on {self.device}")
        self._loaded = True

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a crop image file into a 512-dim vector."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        return features.cpu().numpy().flatten()

    def embed_image_pil(self, pil_img: Image.Image) -> np.ndarray:
        """Embed a PIL image directly."""
        tensor = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        return features.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query into a 512-dim vector."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features.cpu().numpy().flatten()


# Global singleton
embedder = CLIPEmbedder()
