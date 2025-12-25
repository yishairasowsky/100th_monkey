import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# ---------- CANVAS ----------
W, H = 1200, 350
MAX_FONT_SIZE = 260
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ---------- AUTO-FIT RENDER ----------
def render(word):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    padding = 60
    for size in range(MAX_FONT_SIZE, 40, -4):
        f = ImageFont.truetype(FONT_PATH, size)
        bbox = d.textbbox((0, 0), word, font=f)
        if (bbox[2] - bbox[0]) < (W - padding):
            break

    x = (W - (bbox[2] - bbox[0])) // 2
    y = (H - (bbox[3] - bbox[1])) // 2
    d.text((x, y), word, 0, font=f)

    return np.array(img) / 255.0

# ---------- IMAGE GENERATION ----------
def generate_image(near_word, far_word, near_fade, far_clarity, far_blur):
    near = render(near_word)
    far  = render(far_word)

    # ---- NEAR WORD (edge-only, fades with distance) ----
    near_edges = near - gaussian_filter(near, 2.5)
    near_edges_far = gaussian_filter(near_edges, 6.0)
    near_component = (1 - near_fade) * near_edges + near_fade * near_edges_far

    # ---- FAR WORD (mid-scale structure survives blur) ----
    far_base = gaussian_filter(far, far_blur)
    far_mid  = gaussian_filter(far, 4) - gaussian_filter(far, 10)
    far_mid  = np.clip(far_mid, 0, 1)
    far_component = (1 - far_clarity) * far_base + far_clarity * far_mid

    img = near_component + far_component
    img = np.clip(1 - img, 0, 1)

    return Image.fromarray((img * 255).astype("uint8"))

# ---------- STREAMLIT UI ----------
st.set_page_config(layout="wide")
st.title("100th Monkey – Perceptual Word Generator")

near_word = st.text_input("Near word (seen close)", "Country")
far_word  = st.text_input("Far word (seen from afar)", "Magicians")

st.markdown("### Perception controls")

near_fade = st.slider(
    "Near word fade with distance",
    0.0, 1.0, 0.55, 0.05
)

far_clarity = st.slider(
    "Far word clarity at distance",
    0.0, 1.0, 0.55, 0.05
)

far_blur = st.slider(
    "Far distance blur scale",
    6.0, 20.0, 12.0, 1.0
)

image = generate_image(
    near_word,
    far_word,
    near_fade,
    far_clarity,
    far_blur
)

st.image(image, use_column_width=True)
