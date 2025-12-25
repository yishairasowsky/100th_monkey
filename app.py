import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# ================== PAGE ==================
st.set_page_config(layout="wide")

# ================== CANVAS ==================
W, H = 1400, 360
BG_GRAY = 0.9

# ================== BACKGROUND ==================
BG_CLOUD_STRENGTH = 0.15
BG_SCALES = [20.0, 60.0, 160.0]
BG_WEIGHTS = [0.4, 0.35, 0.25]

# ================== FONT ==================
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ================== TEXT RENDER ==================
def render(word):
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    max_size = int(H * 0.85)
    padding = int(W * 0.08)

    for size in range(max_size, 20, -4):
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < (W - padding):
            break

    x = (W - w) // 2
    y = (H - h) // 2
    draw.text((x, y), word, 0, font=font)

    return np.array(img) / 255.0

# ================== IMAGE GENERATION ==================
def generate_image(near_word, far_word, preset):
    # ---- PRESETS ----
    if preset == "Balanced (recommended)":
        NEAR_EDGE_STRENGTH = 0.75
        NEAR_DISTANCE_BLUR = 5.0
        FAR_BLUR_SIGMA = 14.0
        FAR_INK_STRENGTH = 0.8
        FAR_DETAIL_BOOST = 0.3
    elif preset == "Far emphasis":
        NEAR_EDGE_STRENGTH = 0.6
        NEAR_DISTANCE_BLUR = 6.5
        FAR_BLUR_SIGMA = 12.0
        FAR_INK_STRENGTH = 0.95
        FAR_DETAIL_BOOST = 0.35
    elif preset == "Close emphasis":
        NEAR_EDGE_STRENGTH = 0.85
        NEAR_DISTANCE_BLUR = 4.0
        FAR_BLUR_SIGMA = 18.0
        FAR_INK_STRENGTH = 0.7
        FAR_DETAIL_BOOST = 0.2
    else:  # Dreamy
        NEAR_EDGE_STRENGTH = 0.65
        NEAR_DISTANCE_BLUR = 7.0
        FAR_BLUR_SIGMA = 20.0
        FAR_INK_STRENGTH = 0.6
        FAR_DETAIL_BOOST = 0.15

    # ---- RENDER TEXT ----
    near_img = render(near_word)
    far_img = render(far_word)

    # ---- BACKGROUND ----
    noise = np.random.randn(H, W)
    clouds = np.zeros_like(noise)
    for w, s in zip(BG_WEIGHTS, BG_SCALES):
        clouds += w * gaussian_filter(noise, s)
    clouds /= np.std(clouds)
    clouds = np.tanh(clouds * 0.9)
    bg = np.clip(BG_GRAY + BG_CLOUD_STRENGTH * clouds, 0, 1)

    # ---- NEAR WORD (EDGE-PRESERVING) ----
    near_edges = np.clip(
        near_img - gaussian_filter(near_img, 2.0),
        0, 1
    )
    near_edges_blurred = gaussian_filter(near_edges, NEAR_DISTANCE_BLUR)
    near_ink = NEAR_EDGE_STRENGTH * (
        0.75 * near_edges + 0.25 * near_edges_blurred
    )

    # ---- FAR WORD (MID-SCALE MASS) ----
    far_base = 1 - gaussian_filter(far_img, FAR_BLUR_SIGMA)
    far_mid = gaussian_filter(far_base, 4.0) - gaussian_filter(far_base, 10.0)
    far_mid = np.clip(far_mid, 0, 1)
    far_ink = FAR_INK_STRENGTH * (
        far_base + FAR_DETAIL_BOOST * far_mid
    )

    out = np.clip(bg - (near_ink + far_ink), 0, 1)
    return Image.fromarray((out * 255).astype("uint8"))

# ================== UI ==================
st.markdown("## 100th Monkey")

left, right = st.columns([1, 2])

with left:
    near_word = st.text_input("Near word (seen close)", "Qxprgtsdwxfs")
    far_word = st.text_input("Far word (seen from afar)", "Chana Golda")

    preset = st.radio(
    "Viewing mode",
    [
        "Close emphasis (recommended)",
        "Balanced",
        "Far emphasis",
        "Dreamy"
    ],
    index=0
)

    st.caption("Tip: step back 2–3 meters from your screen.")

with right:
    img = generate_image(near_word, far_word, preset)
    st.image(img, use_column_width=True)
