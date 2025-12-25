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

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ================== TEXT RENDER ==================
def render(word):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    max_size = int(H * 0.85)
    padding = int(W * 0.08)

    for size in range(max_size, 20, -4):
        try:
            f = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            f = ImageFont.load_default()

        bbox = d.textbbox((0, 0), word, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw < (W - padding):
            break

    x = (W - tw) // 2
    y = (H - th) // 2
    d.text((x, y), word, 0, font=f)

    return np.array(img) / 255.0

# ================== IMAGE ==================
def generate_image(near_word, far_word, params):
    near_img = render(near_word)
    far_img  = render(far_word)

    # ---- BACKGROUND ----
    noise = np.random.randn(H, W)
    clouds = np.zeros_like(noise)
    for w, s in zip(BG_WEIGHTS, BG_SCALES):
        clouds += w * gaussian_filter(noise, s)
    clouds /= np.std(clouds)
    clouds = np.tanh(clouds * 0.9)
    bg = np.clip(BG_GRAY + BG_CLOUD_STRENGTH * clouds, 0, 1)

    # ---- NEAR WORD ----
    near_edges = np.clip(
        near_img - gaussian_filter(near_img, 2.0),
        0, 1
    )
    near_edges_blurred = gaussian_filter(
        near_edges, params["near_distance_blur"]
    )

    near_ink = params["near_strength"] * (
        0.75 * near_edges + 0.25 * near_edges_blurred
    )

    # ---- FAR WORD ----
    far_base = 1 - gaussian_filter(far_img, params["far_blur"])
    far_mid = gaussian_filter(far_base, 4.0) - gaussian_filter(far_base, 10.0)
    far_mid = np.clip(far_mid, 0, 1)

    far_ink = params["far_strength"] * (
        far_base + params["far_detail"] * far_mid
    )

    out = np.clip(bg - (near_ink + far_ink), 0, 1)
    return Image.fromarray((out * 255).astype("uint8"))

# ================== UI ==================
st.markdown("## 100th Monkey")

left, right = st.columns([1, 2])

with left:
    near_word = st.text_input("Near word", "Qxprgtsdwxfs")
    far_word  = st.text_input("Far word", "Chana Golda")

    # ---- BASE PRESET (Close emphasis) ----
    params = {
        "near_strength": 0.85,
        "near_distance_blur": 4.5,
        "far_strength": 0.7,
        "far_blur": 18.0,
        "far_detail": 0.2,
    }

    with st.expander("Fine tuning (for exploration)"):
        params["near_strength"] = st.slider(
            "Near word strength", 0.5, 1.0, params["near_strength"], 0.02
        )
        params["near_distance_blur"] = st.slider(
            "Near distance blur", 2.0, 10.0, params["near_distance_blur"], 0.5
        )
        params["far_strength"] = st.slider(
            "Far word strength", 0.5, 1.0, params["far_strength"], 0.02
        )
        params["far_blur"] = st.slider(
            "Far distance blur", 10.0, 22.0, params["far_blur"], 0.5
        )
        params["far_detail"] = st.slider(
            "Far detail boost", 0.0, 0.5, params["far_detail"], 0.05
        )

    st.caption("Tip: step back 2–3 meters from your screen.")

with right:
    img = generate_image(near_word, far_word, params)
    st.image(img, use_column_width=True)
