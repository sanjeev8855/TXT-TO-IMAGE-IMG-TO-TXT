from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image

class ImagePatch:
    def __init__(self, patch_array: np.ndarray, row_idx: int, col_idx: int, box: Tuple[int, int, int, int]):
        self.array = patch_array  # (H, W, 3) normalized float32
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.box = box  # (x1, y1, x2, y2)

class PatchExtractor:
    """
    Module 10: Patch-Level Analysis.
    Extracts overlapping spatial patches from an image for localized forensic anomaly classification.
    """
    def __init__(self, patch_size: int = 64, stride: int = 32):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, np_img: np.ndarray) -> Tuple[List[ImagePatch], int, int]:
        """
        Extracts overlapping patches from an RGB image array (H, W, 3).
        Returns list of patches, total_rows, and total_cols.
        """
        h, w, c = np_img.shape
        patches = []

        y_starts = list(range(0, h - self.patch_size + 1, self.stride))
        if not y_starts or (h - self.patch_size) > y_starts[-1]:
            y_starts.append(max(0, h - self.patch_size))
        y_starts = sorted(list(set(y_starts)))

        x_starts = list(range(0, w - self.patch_size + 1, self.stride))
        if not x_starts or (w - self.patch_size) > x_starts[-1]:
            x_starts.append(max(0, w - self.patch_size))
        x_starts = sorted(list(set(x_starts)))

        n_rows = len(y_starts)
        n_cols = len(x_starts)

        for row_idx, y in enumerate(y_starts):
            for col_idx, x in enumerate(x_starts):
                patch_arr = np_img[y:y + self.patch_size, x:x + self.patch_size, :]
                box = (x, y, x + self.patch_size, y + self.patch_size)
                patches.append(ImagePatch(patch_arr, row_idx, col_idx, box))

        return patches, n_rows, n_cols

patch_extractor = PatchExtractor(patch_size=64, stride=32)
