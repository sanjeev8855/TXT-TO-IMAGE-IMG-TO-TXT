from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from backend.config import config
from backend.forensic.patch_extractor import ImagePatch, patch_extractor

class ForensicDetector:
    """
    Module 9: Forensic AI Detector.
    Strictly independent of the generation pipeline. Receives ONLY the image.
    Uses multi-stream computer vision forensic techniques:
    1. Spatial Noise Residual Analysis (Laplacian / High-pass filter residual)
    2. Frequency Domain (2D FFT) Spectral Anomaly & Upsampling footprint Analysis
    3. Patch-level Texture Discrepancy Scoring
    """
    def __init__(self):
        f_conf = config.forensics
        self.patch_size = f_conf.get("patch_size", 64)
        self.stride = f_conf.get("stride", 32)
        self.model_name = config.models.get("forensic_model", "spatial-frequency-patch-detector-v1")

    def analyze_patch(self, patch: ImagePatch) -> float:
        """
        Calculates an anomaly score in [0.0, 1.0] for a single 64x64 patch.
        Higher score indicates stronger detected forensic evidence consistent with
        generative model upsampling or diffusion noise artifacts.
        """
        arr = patch.array
        # Convert to grayscale for residual & spectral analysis
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        # 1. High-pass noise residual (discrete Laplacian)
        # Real camera sensors have Gaussian/Poisson shot noise with uniform spatial variance.
        # AI models frequently have smoothed mid-frequencies with periodic upsampler grid spikes.
        laplacian_kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)

        # Fast 2D convolution for patch
        h, w = gray.shape
        padded = np.pad(gray, 1, mode="reflect")
        residual = (
            padded[0:h, 1:w+1] + padded[2:h+2, 1:w+1] +
            padded[1:h+1, 0:w] + padded[1:h+1, 2:w+2] - 4.0 * gray
        )
        res_var = float(np.var(residual))
        res_mean = float(np.mean(np.abs(residual)))

        # 2. 2D FFT Frequency Analysis
        # Latent diffusion upsampling leaves high-frequency spectral energy imbalance.
        fft2 = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)

        # High-frequency ring vs center low-frequency energy ratio
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius = np.sqrt(x*x + y*y)
        max_r = np.sqrt(cx*cx + cy*cy)

        high_freq_mask = (radius > 0.45 * max_r)
        low_freq_mask = (radius <= 0.25 * max_r)

        high_energy = np.sum(magnitude[high_freq_mask]) + 1e-7
        low_energy = np.sum(magnitude[low_freq_mask]) + 1e-7
        hf_ratio = high_energy / (low_energy + high_energy)

        # 3. Micro-texture variance
        grad_y = np.abs(np.diff(gray, axis=0))
        grad_x = np.abs(np.diff(gray, axis=1))
        edge_density = float(np.mean(grad_y) + np.mean(grad_x))

        # 4. Multi-stream anomaly score composition
        # Calibrate score between 0.0 and 1.0 using smooth logistic sigmoid functions to prevent flat saturation
        s_res = 1.0 / (1.0 + np.exp(-10.0 * (res_mean - 0.20)))
        s_fft = 1.0 / (1.0 + np.exp(-7.0 * (hf_ratio - 0.40)))
        s_edge = 1.0 / (1.0 + np.exp(-8.0 * (edge_density - 0.20)))

        score = float(0.40 * s_res + 0.35 * s_fft + 0.25 * s_edge)
        return float(np.clip(score, 0.02, 0.98))

    def evaluate_image(self, np_img: np.ndarray) -> Dict[str, Any]:
        """
        Executes patch extraction, patch classification, and extracts global forensic streams.
        Returns patch spatial score matrix, coordinates, and full-image stream representations.
        """
        patches, n_rows, n_cols = patch_extractor.extract_patches(np_img)
        score_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)
        patch_records = []

        for p in patches:
            score = self.analyze_patch(p)
            score_matrix[p.row_idx, p.col_idx] = score
            patch_records.append({
                "row": p.row_idx,
                "col": p.col_idx,
                "box": p.box,
                "score": round(score, 4)
            })

        # Generate full-image FFT Power Spectrum & Noise Residual for research inspection
        h, w, c = np_img.shape
        gray_full = 0.2989 * np_img[:, :, 0] + 0.5870 * np_img[:, :, 1] + 0.1140 * np_img[:, :, 2]

        # Full FFT
        fft2_full = np.fft.fft2(gray_full)
        fft_shift_full = np.fft.fftshift(fft2_full)
        magnitude_full = np.log(1.0 + np.abs(fft_shift_full))
        # Normalize for visualization
        norm_fft = (magnitude_full - np.min(magnitude_full)) / (np.max(magnitude_full) - np.min(magnitude_full) + 1e-7)

        # Full Laplacian Residual
        padded_full = np.pad(gray_full, 1, mode="reflect")
        res_full = np.abs(
            padded_full[0:h, 1:w+1] + padded_full[2:h+2, 1:w+1] +
            padded_full[1:h+1, 0:w] + padded_full[1:h+1, 2:w+2] - 4.0 * gray_full
        )
        norm_res = np.clip(res_full * 5.0, 0.0, 1.0)

        return {
            "score_matrix": score_matrix,
            "patch_records": patch_records,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "total_patches": len(patches),
            "fft_spectrum_array": norm_fft,
            "noise_residual_array": norm_res,
        }

forensic_detector = ForensicDetector()
