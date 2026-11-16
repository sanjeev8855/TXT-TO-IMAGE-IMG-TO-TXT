import uuid
from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np

from backend.config import config

class HeatmapGenerator:
    """
    Module 12: Pseudo-Thermal Heatmap Generator.
    Produces:
    1. Standalone Infrared Pseudo-Thermal Heatmap
    2. Alpha-Blended Heatmap Overlay on the Original Image
    3. Multi-stream visualizations (FFT spectrum & Noise Residual maps)
    """
    def __init__(self):
        f_conf = config.forensics
        self.default_alpha = f_conf.get("heatmap_alpha", 0.45)
        self.heatmaps_dir = config.heatmaps_dir

    def apply_thermal_colormap(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Maps a 2D float array in [0.0, 1.0] to an RGB (H, W, 3) pseudo-thermal colormap:
        0.0: Deep Blue
        0.25: Cyan
        0.50: Green
        0.75: Yellow / Orange
        1.0: Deep Red / White
        """
        val = np.clip(anomaly_map, 0.0, 1.0)
        h, w = val.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Vectorized piecewise linear interpolation across key anchors
        # Anchors:
        # 0.00: [0, 0, 160] (Dark Blue)
        # 0.25: [0, 190, 255] (Cyan)
        # 0.50: [40, 220, 40] (Green)
        # 0.75: [255, 200, 0] (Yellow)
        # 1.00: [255, 20, 20] (Red)

        # Region 0: [0.0, 0.25]
        m0 = (val <= 0.25)
        t0 = val / 0.25
        rgb[m0, 0] = 0
        rgb[m0, 1] = (t0[m0] * 190).astype(np.uint8)
        rgb[m0, 2] = (160 + t0[m0] * 95).astype(np.uint8)

        # Region 1: [0.25, 0.50]
        m1 = (val > 0.25) & (val <= 0.50)
        t1 = (val - 0.25) / 0.25
        rgb[m1, 0] = (t1[m1] * 40).astype(np.uint8)
        rgb[m1, 1] = (190 + t1[m1] * 30).astype(np.uint8)
        rgb[m1, 2] = (255 - t1[m1] * 215).astype(np.uint8)

        # Region 2: [0.50, 0.75]
        m2 = (val > 0.50) & (val <= 0.75)
        t2 = (val - 0.50) / 0.25
        rgb[m2, 0] = (40 + t2[m2] * 215).astype(np.uint8)
        rgb[m2, 1] = (220 - t2[m2] * 20).astype(np.uint8)
        rgb[m2, 2] = (40 - t2[m2] * 40).astype(np.uint8)

        # Region 3: [0.75, 1.00]
        m3 = (val > 0.75)
        t3 = (val - 0.75) / 0.25
        rgb[m3, 0] = 255
        rgb[m3, 1] = (200 - t3[m3] * 180).astype(np.uint8)
        rgb[m3, 2] = (t3[m3] * 20).astype(np.uint8)

        return rgb

    def create_overlay(
        self,
        orig_img: Image.Image,
        heatmap_rgb: np.ndarray,
        alpha: Optional[float] = None
    ) -> Image.Image:
        """Alpha blends the pseudo-thermal heatmap over the original image."""
        a = alpha if alpha is not None else self.default_alpha
        orig_rgb = orig_img.convert("RGB")
        w, h = orig_rgb.size

        # Ensure heatmap matches original image size
        heatmap_img = Image.fromarray(heatmap_rgb).resize((w, h), Image.Resampling.BILINEAR)

        # Blend
        overlay = Image.blend(orig_rgb, heatmap_img, alpha=a)
        return overlay

    def save_forensic_visualizations(
        self,
        orig_img: Image.Image,
        anomaly_map: np.ndarray,
        fft_array: Optional[np.ndarray] = None,
        residual_array: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Saves heatmap, overlay, and optional FFT/residual visualizations to disk.
        Returns URLs and absolute paths.
        """
        prefix = f"forensic_{uuid.uuid4().hex[:10]}"

        # 1. Standalone Heatmap
        heatmap_rgb = self.apply_thermal_colormap(anomaly_map)
        heatmap_img = Image.fromarray(heatmap_rgb)
        hm_path = self.heatmaps_dir / f"{prefix}_heatmap.png"
        heatmap_img.save(hm_path, format="PNG")

        # 2. Heatmap Overlay
        overlay_img = self.create_overlay(orig_img, heatmap_rgb, alpha=alpha)
        ov_path = self.heatmaps_dir / f"{prefix}_overlay.png"
        overlay_img.save(ov_path, format="PNG")

        results = {
            "heatmap_path": str(hm_path),
            "heatmap_url": f"/outputs/heatmaps/{hm_path.name}",
            "overlay_path": str(ov_path),
            "overlay_url": f"/outputs/heatmaps/{ov_path.name}",
        }

        # 3. Optional Stream: 2D FFT Spectrum
        if fft_array is not None:
            fft_u8 = (np.clip(fft_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            fft_img = Image.fromarray(fft_u8, mode="L")
            fft_path = self.heatmaps_dir / f"{prefix}_fft.png"
            fft_img.save(fft_path, format="PNG")
            results["fft_path"] = str(fft_path)
            results["fft_url"] = f"/outputs/heatmaps/{fft_path.name}"

        # 4. Optional Stream: Spatial Noise Residual
        if residual_array is not None:
            res_u8 = (np.clip(residual_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            res_img = Image.fromarray(res_u8, mode="L")
            res_path = self.heatmaps_dir / f"{prefix}_residual.png"
            res_img.save(res_path, format="PNG")
            results["residual_path"] = str(res_path)
            results["residual_url"] = f"/outputs/heatmaps/{res_path.name}"

        return results

heatmap_generator = HeatmapGenerator()
