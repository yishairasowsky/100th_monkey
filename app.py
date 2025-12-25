import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# ---------- SETTINGS ----------
W, H = 1400, 300
FONT_SIZE = 200

# ---------- BASE PERCEPTUAL PARAMETERS (YOUR GOOD DEFAULTS) ----------
BG_GRAY = 0.9

NEAR_EDGE_STRENGTH = 0.7
NEAR_EDGE_BLUR     = 2.0
NEAR_DISTANCE_BLUR = 5.0
NEAR_PERSISTENCE   = 0.75

FAR_BLUR_SIGMA     = 14.0
FAR_INK_STRENGTH   = 0.8
FAR_DETAIL_BOOST   = 0.3
FAR_SUPPRESSION    = 0.6

BG_CLOUD_STRENGTH  = 0.15
BG_SCALES          = [20.0, 60.0, 160.0]
BG_WEIGHTS         = [0.4, 0.35, 0.25]

# ---------- RENDER ----------
def render(word):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    # Start big, scale down until it fits
    max_size = int(H * 0.9)
    min_size = 20
    padding = int(W * 0.08)

    for size in range(max_size, min_size, -4):
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            f = ImageFont.load_default()

        bbox = d.textbbox((0, 0), word, font=f)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w < (W - padding):
            break

    x = (W - text_w) // 2
    y = (H - text_h) // 2
    d.text((x, y), word, 0, font=f)

    return np.array(img) / 255.0

# ---------- IMAGE GENERATION ----------
def generate_image(
    near_word,
    far_word,
    near_strength,
    far_strength,
    far_blur
):
    near_img = render(near_word)
    far_img  = render(far_word)

    # ---- CLOUDY BACKGROUND ----
    noise = np.random.randn(H, W)
    clouds = np.zeros_like(noise)
    for w, s in zip(BG_WEIGHTS, BG_SCALES):
        clouds += w * gaussian_filter(noise, s)
    clouds /= np.std(clouds)
    clouds = np.tanh(clouds * 0.9)
    bg = np.clip(BG_GRAY + BG_CLOUD_STRENGTH * clouds, 0, 1)

    # ---- NEAR WORD ----
    near_edges = np.clip(
        near_img - gaussian_filter(near_img, NEAR_EDGE_BLUR),
        0, 1
    )
    near_edges_blurred = gaussian_filter(near_edges, NEAR_DISTANCE_BLUR)

    near_ink = near_strength * (
        NEAR_PERSISTENCE * near_edges +
        (1 - NEAR_PERSISTENCE) * near_edges_blurred
    )

    # ---- FAR WORD ----
    far_base = 1 - gaussian_filter(far_img, far_blur)
    far_mid = gaussian_filter(far_base, 4.0) - gaussian_filter(far_base, 10.0)
    far_mid = np.clip(far_mid, 0, 1)

    far_ink = far_strength * FAR_SUPPRESSION * (
        far_base + FAR_DETAIL_BOOST * far_mid
    )

    out = np.clip(bg - (near_ink + far_ink), 0, 1)
    return Image.fromarray((out * 255).astype("uint8"))

# ---------- STREAMLIT UI ----------
st.set_page_config(layout="wide")

st.markdown("### 100th Monkey")

near_word = st.text_input("Near word", "Qxprgtsdwxfs")
far_word  = st.text_input("Far word",  "Chana Golda")

with st.expander("Subtle adjustments", expanded=False):
    near_strength = st.slider(
        "Near word strength",
        0.5, 0.9,
        NEAR_EDGE_STRENGTH,
        0.02
    )

    far_strength = st.slider(
        "Far word strength",
        0.6, 1.0,
        FAR_INK_STRENGTH,
        0.02
    )

    far_blur = st.slider(
        "Far distance blur",
        10.0, 18.0,
        FAR_BLUR_SIGMA,
        0.5
    )

img = generate_image(
    near_word,
    far_word,
    near_strength,
    far_strength,
    far_blur
)

st.image(img, use_column_width=True)
