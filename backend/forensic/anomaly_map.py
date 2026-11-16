from typing import Tuple, Optional
import numpy as np
from PIL import Image

class AnomalyMapGenerator:
    """
    Module 11: Anomaly Map Generator.
    Converts coarse discrete patch score matrices into smooth, continuous 2D anomaly probability fields
    matching the original image dimensions.
    """
    def generate_continuous_map(
        self,
        score_matrix: np.ndarray,
        target_shape: Tuple[int, int],
        high_res_residual: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Interpolates 2D score matrix (rows, cols) to (target_height, target_width)
        using smooth bicubic interpolation in 32-bit floating point precision.
        When high_res_residual is provided, fuses macro-patch probabilities with
        pixel-level structural residual details for high-fidelity spatial anomaly localization.
        Applies adaptive dynamic range normalization so the resulting continuous field
        exhibits full pseudo-thermal contrast (avoiding monochromatic flat saturation).
        Returns float32 2D array with values strictly bounded in [0.0, 1.0].
        """
        h_target, w_target = target_shape
        score_f32 = np.clip(score_matrix.astype(np.float32), 0.0, 1.0)

        # 32-bit floating point bicubic resampling to preserve continuous gradient details
        mat_img = Image.fromarray(score_f32, mode="F")
        coarse_map = np.array(
            mat_img.resize((w_target, h_target), resample=Image.Resampling.BICUBIC),
            dtype=np.float32
        )

        if high_res_residual is not None:
            # Resize residual array if dimensions don't match target shape
            if high_res_residual.shape != (h_target, w_target):
                res_f32 = high_res_residual.astype(np.float32)
                res_img = Image.fromarray(res_f32, mode="F").resize((w_target, h_target), resample=Image.Resampling.BILINEAR)
                res_arr = np.array(res_img, dtype=np.float32)
            else:
                res_arr = high_res_residual.astype(np.float32)

            # Robust percentile normalization of micro-structural residuals
            rp_low = float(np.percentile(res_arr, 2))
            rp_high = float(np.percentile(res_arr, 98))
            norm_res = np.clip((res_arr - rp_low) / (rp_high - rp_low + 1e-6), 0.0, 1.0)

            # Multi-scale fusion: 60% macro-region patch probability + 40% micro pixel residual detail
            fused_map = 0.60 * coarse_map + 0.40 * norm_res
        else:
            fused_map = coarse_map

        # Adaptive dynamic range normalization to ensure full pseudo-thermal contrast
        # Prevents flat monochromatic saturation when patch scores are within a narrow band
        p_min = float(np.percentile(fused_map, 2))
        p_max = float(np.percentile(fused_map, 98))
        if p_max - p_min > 0.01:
            stretched_map = np.clip((fused_map - p_min) / (p_max - p_min), 0.0, 1.0)
        else:
            stretched_map = np.clip(fused_map, 0.0, 1.0)

        return stretched_map.astype(np.float32)

anomaly_map_generator = AnomalyMapGenerator()
