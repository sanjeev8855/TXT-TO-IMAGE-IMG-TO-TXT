"""
Knowledge Base Builder for all 13 Academic Domains (Section 11)
Populates:
1. Photography terminology
2. Lighting terminology
3. Composition terminology
4. Camera terminology
5. Color terminology
6. Art styles
7. Illustration styles
8. Cinematic terminology
9. Prompt engineering patterns
10. Negative prompt patterns
11. Image-generation parameters
12. Object descriptions
13. Scene descriptions
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import config

DOCS_DIR = config.rag_docs_dir

COLOR_TERMINOLOGY = [
    {
        "id": "color_001",
        "category": "color_terminology",
        "title": "Complementary and Split-Complementary Harmonies",
        "description": "Colors directly opposite on the color wheel (e.g. amber and deep cyan, or magenta and olive) create maximal visual contrast, dramatic tension, and vibrant optical punch.",
        "keywords": ["complementary colors", "split-complementary", "color contrast", "teal and orange", "chromatic tension"]
    },
    {
        "id": "color_002",
        "category": "color_terminology",
        "title": "Analogous and Monochromatic Color Palettes",
        "description": "Colors adjacent on the color wheel (e.g. forest green, sage, and chartreuse) cultivate serene, organic cohesion and tranquil atmosphere without jarring optical clash.",
        "keywords": ["analogous colors", "monochromatic", "harmonious palette", "subtle gradation", "tonal harmony"]
    },
    {
        "id": "color_003",
        "category": "color_terminology",
        "title": "High-Key vs Low-Key Tonal Distribution",
        "description": "High-key schemes emphasize bright, luminous values, pastel hues, and minimal deep shadows. Low-key schemes rely on predominant dark values, deep chiaroscuro, and selective highlights.",
        "keywords": ["high-key", "low-key", "chiaroscuro", "luminance range", "tonal key", "shadow dominance"]
    },
    {
        "id": "color_004",
        "category": "color_terminology",
        "title": "Warm vs Cool Chromatic Balance",
        "description": "Warm hues (reds, oranges, golds) evoke intimacy, energy, and radiant heat; cool hues (cyans, deep blues, lavenders) establish distance, calm, mystery, and nocturnal depth.",
        "keywords": ["warm colors", "cool colors", "color temperature", "chromatic temperature", "warm cool contrast"]
    }
]

ILLUSTRATION_STYLES = [
    {
        "id": "illust_001",
        "category": "illustration_styles",
        "title": "Cel-Shaded Anime and Graphic Manga",
        "description": "Clean ink outlines with flat or two-step tonal shading, expressive line weights, stylized anatomical proportions, and vibrant animation aesthetic.",
        "keywords": ["cel shading", "anime", "manga", "ink line art", "graphic novel", "flat color"]
    },
    {
        "id": "illust_002",
        "category": "illustration_styles",
        "title": "Wet-on-Wet Botanical Watercolor",
        "description": "Soft fluid pigment bleed, organic color blooms, translucent washes, visible paper grain texture, and delicate spontaneous brush strokes.",
        "keywords": ["watercolor", "wet on wet", "botanical illustration", "fluid wash", "paper texture", "translucent pigment"]
    },
    {
        "id": "illust_003",
        "category": "illustration_styles",
        "title": "Risograph and Screenprint Graphic Art",
        "description": "Limited spot-color inks, intentional misregistration, half-tone dither dot patterns, high tactile vintage paper presence, and bold modernist shapes.",
        "keywords": ["risograph", "screenprint", "halftone dots", "dither pattern", "spot color", "misregistration", "printmaking"]
    }
]

CINEMATIC_TERMINOLOGY = [
    {
        "id": "cine_001",
        "category": "cinematic_terminology",
        "title": "Anamorphic Widescreen and Horizontal Flare",
        "description": "2.39:1 aspect ratio with oval out-of-focus bokeh highlights, subtle horizontal blue streak flares, and gentle barrel optical distortion characteristic of anamorphic glass.",
        "keywords": ["anamorphic", "widescreen", "2.39:1", "horizontal lens flare", "oval bokeh", "cinematic aspect ratio"]
    },
    {
        "id": "cine_002",
        "category": "cinematic_terminology",
        "title": "Atmospheric Volumetric Fog and God Rays",
        "description": "Suspended particulate haze scattering directional lighting into visible sunbeams (crepuscular rays), creating deep three-dimensional spatial layering.",
        "keywords": ["volumetric lighting", "god rays", "crepuscular rays", "atmospheric haze", "mist", "shafts of light"]
    },
    {
        "id": "cine_003",
        "category": "cinematic_terminology",
        "title": "Dutch Angle and Dynamic Canting",
        "description": "Camera tilted deliberately off horizontal axis to introduce psychological tension, disorientation, dynamism, and impending motion in dramatic scenes.",
        "keywords": ["dutch angle", "canted frame", "tilted horizon", "psychological tension", "dynamic composition"]
    }
]

PROMPT_ENGINEERING = [
    {
        "id": "prompt_001",
        "category": "prompt_engineering",
        "title": "Hierarchical Structural Prompt Syntax",
        "description": "Arranging prompts from primary subject entity -> environmental setting -> lighting design -> optical camera specifications -> aesthetic rendering style maximizes diffusion fidelity.",
        "keywords": ["prompt structure", "prompt hierarchy", "subject first", "syntax formatting", "diffusion conditioning"]
    },
    {
        "id": "prompt_002",
        "category": "prompt_engineering",
        "title": "Token Weighting and Emphasis Modifiers",
        "description": "Emphasizing focal entities using structural isolation, specific technical vocabulary, and avoiding contradictory modifiers ensures precise cross-attention guidance.",
        "keywords": ["token weighting", "cross-attention guidance", "modifier balance", "diffusion prompt engineering"]
    }
]

NEGATIVE_PROMPT = [
    {
        "id": "neg_001",
        "category": "negative_prompt",
        "title": "Defensive Negative Prompting Patterns",
        "description": "Standard negative conditioning vectors to suppress common generative flaws: oversaturated highlights, chromatic aberration, plastic skin, anatomical asymmetry, watermarks, and cropped boundaries.",
        "keywords": ["negative prompt", "artifact suppression", "oversaturation", "bad anatomy", "watermark avoidance"]
    },
    {
        "id": "neg_002",
        "category": "negative_prompt",
        "title": "Boundary and Text Artifact Filtering",
        "description": "Explicit exclusion of synthetic typographical stamps, blurred foreground clutter, JPEG compression artifacts, and duplicate limb anomalies.",
        "keywords": ["text avoidance", "clean edges", "compression artifact suppression", "duplicate limbs"]
    }
]

GENERATION_PARAMETERS = [
    {
        "id": "param_001",
        "category": "generation_parameters",
        "title": "Classifier-Free Guidance (CFG Scale)",
        "description": "CFG scale (typically 6.0 to 8.5) balances prompt adherence against image naturalness. Too low causes drift from prompt; too high creates oversaturated contrast and burnt edges.",
        "keywords": ["cfg scale", "guidance scale", "classifier-free guidance", "prompt adherence", "saturation balance"]
    },
    {
        "id": "param_002",
        "category": "generation_parameters",
        "title": "Diffusion Schedulers (DDIM, DPM++, Euler)",
        "description": "Higher-order multistep solvers like DPM++ 2M Karras achieve convergence in 20-30 steps, while deterministic DDIM schedules ensure exact mathematical reproducibility given identical seeds.",
        "keywords": ["scheduler", "DDIM", "DPM++", "Euler a", "sampling steps", "reproducibility"]
    }
]

OBJECT_DESCRIPTIONS = [
    {
        "id": "obj_001",
        "category": "object_descriptions",
        "title": "Tactile Material Properties and Specular Highlights",
        "description": "Detailed description of surface micro-roughness, brushed aluminum reflection, weathered porous stone, polished hardwood grain, and delicate velvet sheen.",
        "keywords": ["material description", "tactile texture", "specular highlight", "surface roughness", "micro-texture"]
    },
    {
        "id": "obj_002",
        "category": "object_descriptions",
        "title": "Organic Biological and Foliage Morphology",
        "description": "Natural cellular structures, translucent leaf sub-surface scattering, layered pine needles, bark fissures, and organic botanical variation.",
        "keywords": ["foliage", "botanical detail", "subsurface scattering", "organic morphology", "bark texture"]
    }
]

SCENE_DESCRIPTIONS = [
    {
        "id": "scene_001",
        "category": "scene_descriptions",
        "title": "Atmospheric Coastal Twilight and Ocean Horizon",
        "description": "Expansive sea vista during astronomical twilight, rhythmic wave crests catching pastel horizon reflection, damp wet sand mirroring gradient sky tones.",
        "keywords": ["coastal sunset", "ocean horizon", "seascape", "golden hour", "wet sand reflection", "twilight"]
    },
    {
        "id": "scene_002",
        "category": "scene_descriptions",
        "title": "Futuristic Cybernetic Urban Streetscape",
        "description": "Rain-slicked asphalt reflecting vertical neon signage, dense architectural tiers, holographic advertisements, and shadowy pedestrian silhouettes in dense steam.",
        "keywords": ["cyberpunk street", "neon reflections", "rain-slicked asphalt", "futuristic city", "urban noir"]
    }
]

ALL_NEW_DOMAINS = [
    ("color_terminology.json", COLOR_TERMINOLOGY),
    ("illustration_styles.json", ILLUSTRATION_STYLES),
    ("cinematic_terminology.json", CINEMATIC_TERMINOLOGY),
    ("prompt_engineering.json", PROMPT_ENGINEERING),
    ("negative_prompt.json", NEGATIVE_PROMPT),
    ("generation_parameters.json", GENERATION_PARAMETERS),
    ("object_descriptions.json", OBJECT_DESCRIPTIONS),
    ("scene_descriptions.json", SCENE_DESCRIPTIONS),
]

def ensure_all_domains():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    count_new = 0
    for filename, docs in ALL_NEW_DOMAINS:
        file_path = DOCS_DIR / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(docs, f, indent=2)
            count_new += 1
    return count_new

if __name__ == "__main__":
    n = ensure_all_domains()
    print(f"Ensured all 13 domains. Added {n} new domain corpora files.")
