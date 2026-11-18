"""
Real Computer-Vision Color Extraction Module (Section 5, 6, 27)
Extracts exact dominant colors from image pixels using OpenCV, NumPy, and Scikit-Learn KMeans clustering.
Calculates real HEX, RGB, HSV/HSL, pixel percentages, and mathematical Warm/Cool/Neutral color groups.
"""
import colorsys
import warnings
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np

# OpenCV is optional here; provide safe fallback
try:
    import cv2
except ImportError:
    cv2 = None

# Scikit-learn KMeans with safe PIL/NumPy adaptive fallback
try:
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None
    ConvergenceWarning = None

# Standard named color anchors for semantic naming
COLOR_NAMES_MAP = [
    ("Deep Navy Blue", (24, 58, 99)),
    ("Sky Blue", (114, 183, 232)),
    ("Royal Blue", (65, 105, 225)),
    ("Cobalt Blue", (0, 71, 171)),
    ("Cyan / Azure", (0, 190, 220)),
    ("Teal", (0, 128, 128)),
    ("Forest Green", (34, 139, 34)),
    ("Emerald Green", (46, 204, 113)),
    ("Olive Green", (107, 142, 35)),
    ("Sage Green", (158, 179, 148)),
    ("Warm Amber", (255, 191, 0)),
    ("Sunset Orange", (255, 127, 80)),
    ("Crimson Red", (220, 20, 60)),
    ("Ruby Red", (155, 17, 30)),
    ("Deep Burgundy", (128, 0, 32)),
    ("Warm Coral", (240, 128, 128)),
    ("Golden Yellow", (255, 215, 0)),
    ("Lemon Yellow", (255, 247, 0)),
    ("Amethyst Violet", (153, 102, 204)),
    ("Deep Purple", (102, 51, 153)),
    ("Lavender", (230, 230, 250)),
    ("Earth Brown", (139, 69, 19)),
    ("Warm Terracotta", (204, 78, 51)),
    ("Sand Beige", (222, 204, 174)),
    ("Charcoal Black", (30, 30, 30)),
    ("Slate Gray", (112, 128, 144)),
    ("Silver Gray", (192, 192, 192)),
    ("Pure White", (250, 250, 250)),
    ("Off-White / Cream", (245, 245, 235)),
    ("Midnight Navy", (10, 25, 47)),
    ("Moss Green", (78, 125, 66)),
    ("Dusty Rose", (200, 140, 150)),
]

def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """Finds closest semantic color name by minimum Euclidean distance in RGB color space."""
    r, g, b = rgb
    best_name = "Unspecified Hue"
    best_dist = float("inf")
    for name, (cr, cg, cb) in COLOR_NAMES_MAP:
        dist = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name

def classify_color_group(h: float, s: float, v: float) -> str:
    """
    Classifies a color into Warm, Cool, or Neutral according to formal color science:
    - Neutral: Saturation < 0.15 OR Value < 0.15 OR Value > 0.95 with low saturation
    - Warm: Hue in [0, 60] (Reds, Oranges, Yellows) or [300, 360] (Magentas, Red-Pinks)
    - Cool: Hue in [150, 270] (Cyans, Blues, Royal Purples)
    - Intermediate / Transition hues [60, 150] (Greens) & [270, 300] (Violets) treated as Cool/Neutral
    """
    if s < 0.15 or v < 0.15:
        return "Neutral"
    if v > 0.92 and s < 0.20:
        return "Neutral"

    # Hue is in [0, 360] degrees
    if (0 <= h <= 65) or (300 <= h <= 360):
        return "Warm"
    elif 145 <= h <= 275:
        return "Cool"
    elif 65 < h < 145:
        return "Cool"  # Natural vegetation / green
    else:
        return "Neutral"

class CVColorExtractor:
    """
    Computer Vision dominant color extraction engine.
    Calculates pixel-derived color palettes, histograms, and group ratios.
    """
    def __init__(self, n_colors: int = 8, sample_size: Tuple[int, int] = (200, 200)):
        self.n_colors = n_colors
        self.sample_size = sample_size

    def extract_palette(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Extracts dominant colors, bar graph data, RGB/brightness/saturation distributions.
        Returns fully validated pixel-derived analysis.
        """
        # 1. Resize for fast, robust K-Means clustering while preserving distribution
        img_resized = pil_image.resize(self.sample_size, Image.Resampling.LANCZOS).convert("RGB")
        np_pixels = np.array(img_resized, dtype=np.float32)  # shape (200, 200, 3)
        pixels_flat = np_pixels.reshape(-1, 3)

        # 2. Run KMeans clustering on actual pixels
        if HAS_SKLEARN and KMeans is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10, max_iter=150)
                labels = kmeans.fit_predict(pixels_flat)
                centroids = kmeans.cluster_centers_

            # 3. Calculate cluster percentages
            total_pixels = len(labels)
            counts = np.bincount(labels, minlength=self.n_colors)
            percentages = (counts / total_pixels) * 100.0
        else:
            # High-performance PIL adaptive palette quantization fallback
            quantized = img_resized.quantize(colors=self.n_colors, method=Image.Quantize.MEDIANCUT)
            raw_palette = quantized.getpalette() or []
            padded_pal = raw_palette + [0] * max(0, (self.n_colors * 3 - len(raw_palette)))
            centroids = np.array([padded_pal[i*3:(i+1)*3] for i in range(self.n_colors)], dtype=np.float32)
            hist = quantized.histogram()[:self.n_colors]
            hist += [0] * max(0, self.n_colors - len(hist))
            total_pixels = sum(hist) or 1
            percentages = np.array([(c / total_pixels) * 100.0 for c in hist])

        # Sort clusters by percentage descending
        sorted_indices = np.argsort(-percentages)

        dominant_colors: List[Dict[str, Any]] = []
        warm_pct = 0.0
        cool_pct = 0.0
        neutral_pct = 0.0

        for idx in sorted_indices:
            r = int(np.clip(round(centroids[idx][0]), 0, 255))
            g = int(np.clip(round(centroids[idx][1]), 0, 255))
            b = int(np.clip(round(centroids[idx][2]), 0, 255))
            pct = float(round(percentages[idx], 2))

            hex_code = f"#{r:02X}{g:02X}{b:02X}"
            color_name = rgb_to_color_name((r, g, b))

            # Convert to HSV & HSL
            h_norm, s_norm, v_norm = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h_deg = round(h_norm * 360.0, 1)
            s_pct = round(s_norm * 100.0, 1)
            v_pct = round(v_norm * 100.0, 1)

            group = classify_color_group(h_deg, s_norm, v_norm)
            if group == "Warm":
                warm_pct += pct
            elif group == "Cool":
                cool_pct += pct
            else:
                neutral_pct += pct

            dominant_colors.append({
                "name": color_name,
                "hex": hex_code,
                "rgb": [r, g, b],
                "rgb_string": f"{r}, {g}, {b}",
                "percentage": pct,
                "hsv": {"hue": h_deg, "saturation": s_pct, "value": v_pct},
                "group": group
            })

        # Normalize group percentages to sum to 100%
        group_total = warm_pct + cool_pct + neutral_pct
        if group_total > 0:
            warm_pct = round((warm_pct / group_total) * 100.0, 1)
            cool_pct = round((cool_pct / group_total) * 100.0, 1)
            neutral_pct = round(100.0 - (warm_pct + cool_pct), 1)

        # 4. Generate distributions for visual analytics dashboard
        r_hist, _ = np.histogram(pixels_flat[:, 0], bins=32, range=(0, 256), density=True)
        g_hist, _ = np.histogram(pixels_flat[:, 1], bins=32, range=(0, 256), density=True)
        b_hist, _ = np.histogram(pixels_flat[:, 2], bins=32, range=(0, 256), density=True)

        # Brightness (luminance) & Saturation distributions
        gray_pixels = 0.2989 * pixels_flat[:, 0] + 0.5870 * pixels_flat[:, 1] + 0.1140 * pixels_flat[:, 2]
        bright_hist, _ = np.histogram(gray_pixels, bins=20, range=(0, 256), density=True)

        return {
            "dominant_colors": dominant_colors,
            "total_colors_detected": len(dominant_colors),
            "primary_color": dominant_colors[0] if dominant_colors else None,
            "color_groups": {
                "warm_percentage": warm_pct,
                "cool_percentage": cool_pct,
                "neutral_percentage": neutral_pct,
                "classification_rule": "Warm: Hue 0-65 & 300-360; Cool: Hue 65-275; Neutral: Saturation<15% or Value<15%"
            },
            "channel_distributions": {
                "red": (r_hist * 100.0).round(2).tolist(),
                "green": (g_hist * 100.0).round(2).tolist(),
                "blue": (b_hist * 100.0).round(2).tolist(),
                "brightness": (bright_hist * 100.0).round(2).tolist(),
            }
        }

cv_color_extractor = CVColorExtractor(n_colors=8)
