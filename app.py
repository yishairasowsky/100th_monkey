import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# ---------- CANVAS ----------
W, H = 1200, 350
FONT_SIZE = 260

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ---------- RENDER ----------
def render(word):
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    # --- auto-fit font size ---
    max_font = FONT_SIZE
    min_font = 40
    padding = 60

    for size in range(max_font, min_font, -4):
        f = ImageFont.truetype(FONT_PATH, size)
        bbox = d.textbbox((0, 0), word, font=f)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w < W - padding:
            break

    x = (W - text_w) // 2
    y = (H - text_h) // 2
    d.text((x, y), word, 0, font=f)

    return np.array(img) / 255.0

def generate_image(near_word, far_word):
    near = render(near_word)
    far  = render(far_word)

    near_detail = near - gaussian_filter(near, 3)
    far_mass    = gaussian_filter(far, 22)

    img = 0.7 * near_detail + 0.3 * far_mass
    img = np.clip(1 - img, 0, 1)

    return Image.fromarray((img * 255).astype("uint8"))

# ---------- STREAMLIT UI ----------
st.set_page_config(layout="wide")
st.title("100th Monkey Generator")

near_word = st.text_input("Near word (seen close)", "Month")
far_word  = st.text_input("Far word (seen from afar)", "March")

image = generate_image(near_word, far_word)
st.image(image, use_column_width=True)
