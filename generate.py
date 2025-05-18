import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.draw import line_aa, ellipse_perimeter
from skimage.transform import resize
from math import atan2
from io import BytesIO
from PIL import Image
import tempfile

# Utilities
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def largest_square(image):
    short_edge = min(image.shape[:2])
    center = (image.shape[0] // 2, image.shape[1] // 2)
    half = short_edge // 2
    return image[center[0]-half:center[0]+half, center[1]-half:center[1]+half]

def create_circle_nail_positions(shape, nail_step=4, r1=1, r2=1):
    h, w = shape
    center = (h // 2, w // 2)
    radius = min(h, w) // 2 - 1
    rr, cc = ellipse_perimeter(center[0], center[1], int(radius*r1), int(radius*r2))
    nails = list(set(zip(rr, cc)))
    nails.sort(key=lambda c: atan2(c[0] - center[0], c[1] - center[1]))
    return np.array(nails[::nail_step])

def init_canvas(shape, white=True):
    return np.ones(shape) if white else np.zeros(shape)

def get_aa_line(p1, p2, strength, canvas):
    rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])
    rr = np.clip(rr, 0, canvas.shape[0]-1)
    cc = np.clip(cc, 0, canvas.shape[1]-1)
    overlay = canvas[rr, cc] + strength * val
    return np.clip(overlay, 0, 1), rr, cc

def find_best_nail(current, nails, str_canvas, orig, strength):
    best_score = -np.inf
    best_idx = -1
    best_pos = None

    for idx, nail in enumerate(nails):
        overlay, rr, cc = get_aa_line(current, nail, strength, str_canvas)
        before = np.abs(str_canvas[rr, cc] - orig[rr, cc])**2
        after = np.abs(overlay - orig[rr, cc])**2
        score = np.sum(before - after)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_pos = nail

    return best_idx, best_pos, best_score

def create_art(nails, orig_img, str_canvas, strength, limit):
    current = nails[0]
    order = [0]
    fails = 0
    i = 0
    while i < limit:
        i += 1
        idx, pos, score = find_best_nail(current, nails, str_canvas, orig_img, strength)
        if score <= 0:
            fails += 1
            if fails >= 5:
                break
            continue
        overlay, rr, cc = get_aa_line(current, pos, strength, str_canvas)
        str_canvas[rr, cc] = overlay
        current = pos
        order.append(idx)
    return order

def scale_nails(xr, yr, nails):
    return [(int(y*n[0]), int(x*n[1])) for n in nails]

def draw_pull_order(order, canvas, nails, strength):
    for i in range(len(order)-1):
        p1, p2 = nails[order[i]], nails[order[i+1]]
        rr, cc, val = line_aa(p1[0], p1[1], p2[0], p2[1])
        rr = np.clip(rr, 0, canvas.shape[0]-1)
        cc = np.clip(cc, 0, canvas.shape[1]-1)
        canvas[rr, cc] += val * strength
    return np.clip(canvas, 0, 1)

# Streamlit Interface
st.title("🧵 String Art Generator – Black Thread on White Canvas")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
nail_step = st.slider("Nail step (affects spacing)", 2, 10, 4)
num_lines = st.slider("Number of lines", 200, 5000, 1000, step=100)

if uploaded:
    # Load image
    image = Image.open(uploaded).convert("RGB")
    img_np = np.asarray(image) / 255.0
    gray = rgb2gray(img_np)

    # Preprocess
    gray = largest_square(gray)
    gray = resize(gray, (300, 300))
    nails = create_circle_nail_positions(gray.shape, nail_step)
    orig = gray * 0.9
    canvas = init_canvas(gray.shape, white=True)

    st.text(f"Total nails: {len(nails)}")
    order = create_art(nails, orig, canvas, strength=-0.05, limit=num_lines)

    scaled_nails = scale_nails(1, 1, nails)
    final_canvas = init_canvas(gray.shape, white=True)
    result = draw_pull_order(order, final_canvas, scaled_nails, strength=-0.1)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(result, cmap="gray", vmin=0, vmax=1)
    ax.axis('off')
    st.pyplot(fig)

    # Downloadable output
    img_out = (result * 255).astype(np.uint8)
    output = Image.fromarray(img_out)
    buf = BytesIO()
    output.save(buf, format="PNG")
    st.download_button("Download Result", buf.getvalue(), "string_art.png", "image/png")

    # Show nail sequence
    st.text_area("Nail Pull Order", "-".join(str(i) for i in order), height=150)
