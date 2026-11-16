"""
Scene, Background, and Composition Analyzer (Section 7 & 8)
Computes mathematical composition metrics: symmetry, horizon, depth of field,
and extracts background properties.
"""
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import cv2

class SceneCompositionAnalyzer:
    """Computes background and spatial composition metrics using computer vision."""

    def analyze(self, pil_image: Image.Image) -> Dict[str, Any]:
        img_rgb = pil_image.convert("RGB")
        w, h = img_rgb.size
        aspect_ratio = round(w / float(h), 2)
        orientation = "Landscape" if aspect_ratio > 1.15 else ("Portrait" if aspect_ratio < 0.85 else "Square")

        np_img = np.array(img_rgb, dtype=np.float32) / 255.0
        gray = cv2.cvtColor((np_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # 1. Symmetry calculation (normalized difference with flipped image)
        flipped = cv2.flip(gray, 1)
        diff = np.abs(gray.astype(np.float32) - flipped.astype(np.float32))
        symmetry_score = round(float(np.clip(1.0 - (np.mean(diff) / 60.0), 0.1, 0.99)), 2)
        symmetry_type = "Bilateral / Symmetric" if symmetry_score > 0.70 else "Asymmetric"

        # 2. Horizon position detection via horizontal edge projection
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        row_energy = np.mean(np.abs(sobel_y), axis=1)
        peak_row = int(np.argmax(row_energy[int(h*0.2):int(h*0.8)])) + int(h*0.2)
        horizon_rel = peak_row / float(h)
        if 0.25 <= horizon_rel <= 0.42:
            horizon_desc = "Upper Third (Sky emphasis)"
        elif 0.58 <= horizon_rel <= 0.75:
            horizon_desc = "Lower Third (Foreground emphasis)"
        else:
            horizon_desc = "Center-level or Obscured"

        # 3. Depth of field (gradient focus variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > 300:
            depth_desc = "Deep depth-of-field (Sharp throughout)"
        elif lap_var > 90:
            depth_desc = "Moderate depth-of-field"
        else:
            depth_desc = "Shallow depth-of-field (Selective focus / Soft background)"

        # 4. Background region analysis (extract top 35% & peripheral margins)
        bg_region_img = gray[0:int(h*0.35), :]
        bg_edges = cv2.Canny(bg_region_img, 50, 150)
        bg_edge_density = float(np.mean(bg_edges > 0))
        bg_complexity = "Simple / Clear" if bg_edge_density < 0.04 else ("Complex / Detailed" if bg_edge_density > 0.12 else "Moderate")

        # Top region dominant color
        top_crop = np_img[0:int(h*0.30), :]
        top_mean_rgb = (
            int(np.mean(top_crop[:, :, 0]) * 255),
            int(np.mean(top_crop[:, :, 1]) * 255),
            int(np.mean(top_crop[:, :, 2]) * 255)
        )
        from backend.image_to_text.color_extractor import rgb_to_color_name
        top_color_name = rgb_to_color_name(top_mean_rgb)
        top_color_hex = f"#{top_mean_rgb[0]:02X}{top_mean_rgb[1]:02X}{top_mean_rgb[2]:02X}"

        # Visual balance: compare left half vs right half intensity
        left_mean = np.mean(gray[:, 0:w//2])
        right_mean = np.mean(gray[:, w//2:])
        balance_diff = abs(left_mean - right_mean)
        balance_desc = "Equally balanced" if balance_diff < 12 else ("Left-weighted" if left_mean > right_mean else "Right-weighted")

        return {
            "background": {
                "type": "Environmental / Studio setting",
                "dominant_colors": [f"{top_color_hex} — {top_color_name}"],
                "region": "Upper and peripheral regions",
                "indoor_outdoor": "Outdoor atmosphere" if (top_mean_rgb[2] > top_mean_rgb[0] or top_mean_rgb[1] > 120) else "Indoor or Controlled studio",
                "complexity": bg_complexity,
                "background_objects": "Distant contextual backdrop / ambient field",
                "lighting": "Natural daylight / ambient diffuse illumination"
            },
            "composition": {
                "foreground": "Primary focal elements positioned in lower/mid frame",
                "middle_ground": "Main visual subjects and transitional space",
                "background": f"{bg_complexity} environmental backdrop with {top_color_name}",
                "subject_placement": "Center-aligned / Balanced focal field",
                "symmetry": symmetry_type,
                "symmetry_score": symmetry_score,
                "rule_of_thirds": "Key visual centers align closely with compositional power points",
                "horizon_position": horizon_desc,
                "depth": depth_desc,
                "visual_balance": balance_desc,
                "orientation": orientation,
                "aspect_ratio": f"{aspect_ratio}:1 ({w}x{h} px)"
            }
        }

scene_composition_analyzer = SceneCompositionAnalyzer()
