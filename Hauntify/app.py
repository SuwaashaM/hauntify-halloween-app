import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pickle
import random

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Hauntify 🎃", page_icon="👻", layout="wide")

# Halloween-themed CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom, #1b1b2f, #2b0b3b);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    font-family: 'Creepster', cursive;
    text-shadow: 2px 2px 8px #ff6a00;
}
.stButton>button {
    background: linear-gradient(45deg, #ff6600, #ffcc00);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5em 1em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;'>🕸️ Hauntify: Halloween Filters 🎃</h1>
<p style='text-align:center;color:lightgrey;'>Capture your photo and apply spooky, fun effects instantly!</p>
""", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open('models/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except:
        st.warning("Text model not found. Text analysis disabled.")
        return None, None

model, vectorizer = load_model()

# -----------------------------
# Mappings
# -----------------------------
label_map = {0: "Not Spooky 🕯️", 1: "Spooky 👻", 2: "Very Spooky 💀"}
costume_map = {
    "Not Spooky 🕯️": ["🎃 Pumpkin Costume", "🐈 Cat Costume", "👻 Friendly Ghost Costume"],
    "Spooky 👻": ["🧙‍♀️ Witch Costume", "💀 Skeleton Costume", "🕸️ Phantom Costume"],
    "Very Spooky 💀": ["🧛 Vampire Costume", "🧟 Zombie Costume", "🤡 Creepy Clown Costume"]
}

filter_comments = {
    "Vampire": ("🧛 A pale and mysterious look! Sharp and elegant.", 4),
    "Ghost": ("👻 Faded and mystical. Looks ethereal.", 3),
    "Zombie": ("🧟‍♂️ Terrifying with brown wavy stripes! Very creepy.", 5),
    "Witch": ("🧙 Magical with sparkling stars! Stunning Halloween vibe.", 4),
    "None": ("🌟 Your original picture is lovely!", 5)
}

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

# -----------------------------
# LEFT: Spookiness Detector
# -----------------------------
with col1:
    st.subheader("📝 Spookiness Detector & Costume Suggestions")
    user_input = st.text_area(
        "Enter your Halloween-themed sentence:",
        height=150,
        placeholder="Example: The graveyard whispers under the blood moon..."
    )

    spooky_score = 0
    spooky_label = "Not Spooky 🕯️"
    if st.button("🎃 Analyze My Text"):
        if user_input.strip():
            if model is None or vectorizer is None:
                st.warning("Text model not available.")
            else:
                features = vectorizer.transform([user_input])
                prediction = model.predict(features)[0]
                spooky_label = label_map.get(prediction, "Unknown")
                st.success(f"🕷️ Spookiness Level: **{spooky_label}**")

                if hasattr(model, "predict_proba"):
                    spooky_score = model.predict_proba(features)[0][prediction] * 100
                    st.caption(f"Spookiness Confidence: {spooky_score:.2f}%")
                else:
                    spooky_score = 50

                st.markdown("### 👗 Recommended Costumes:")
                for costume in costume_map.get(spooky_label, []):
                    st.markdown(f"<div style='padding:5px; margin:3px; border:1px dashed #ff6600; border-radius:10px; transition: all 0.3s ease;'>{costume}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a sentence first!")

# -----------------------------
# Wavy & Sparkle Effects
# -----------------------------
def apply_wavy_effect(img_cv, amplitude=5, frequency=30):
    h, w, c = img_cv.shape
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x = map_x + amplitude * np.sin(2 * np.pi * map_y / frequency)
    wavy_img = cv2.remap(img_cv, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return wavy_img

def add_sparkles(img_pil, count=50):
    draw = ImageDraw.Draw(img_pil)
    for _ in range(count):
        x = random.randint(0, img_pil.width-1)
        y = random.randint(0, img_pil.height-1)
        size = random.randint(2, 6)
        color = (255, 255, random.randint(180, 255))
        draw.ellipse((x, y, x+size, y+size), fill=color)
    return img_pil

# -----------------------------
# RIGHT: Camera & Filters
# -----------------------------
with col2:
    st.subheader("🪞 Capture Photo & Apply Filter")
    captured_file = st.camera_input("📷 Capture your photo")

    intensity = st.slider("Adjust Spooky Intensity", 0.5, 1.5, 1.0, 0.1)

    if captured_file:
        image = Image.open(captured_file).convert("RGB")
        st.image(image, caption="Original", use_container_width=True)

        filter_choice = st.selectbox(
            "🎭 Choose Your Spooky Look:",
            ["None", "Vampire", "Ghost", "Zombie", "Witch"]
        )

        img_cv = np.array(image)
        opacity = 0.6 * intensity

        if filter_choice == "Vampire":
            img = cv2.convertScaleAbs(img_cv, alpha=1.1*intensity, beta=-40)
            img[:, :, 0] = cv2.add(img[:, :, 0], int(20*intensity))
            img[:, :, 2] = cv2.add(img[:, :, 2], int(30*intensity))
            img = apply_wavy_effect(img, amplitude=4, frequency=25)
            result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = ImageEnhance.Contrast(result).enhance(1.3 * intensity)

        elif filter_choice == "Ghost":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            ghost = cv2.merge([gray, gray, gray])
            overlay = np.full_like(ghost, 255)
            img = cv2.addWeighted(ghost, 0.8, overlay, 0.2, 0)
            img = apply_wavy_effect(img, amplitude=6, frequency=20)
            result = Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.5))

        elif filter_choice == "Zombie":
            zombie = cv2.convertScaleAbs(img_cv, alpha=0.9, beta=-20)
            h, w, c = zombie.shape
            overlay = np.zeros_like(zombie, dtype=np.uint8)
            for y in range(0, h, 20):
                amplitude = 5
                frequency = 30
                for x in range(w):
                    offset = int(amplitude * np.sin(2 * np.pi * y / frequency))
                    yy = min(h-1, max(0, y + offset))
                    overlay[yy, x] = [50, 30, 0]
            img = cv2.addWeighted(zombie, 0.7, overlay, 0.3, 0)
            img = apply_wavy_effect(img, amplitude=3, frequency=40)
            result = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        elif filter_choice == "Witch":
            witch = cv2.convertScaleAbs(img_cv, alpha=0.8, beta=-20)
            witch[:, :, 0] = cv2.add(witch[:, :, 0], int(30*intensity))
            witch[:, :, 1] = cv2.add(witch[:, :, 1], int(10*intensity))
            witch[:, :, 2] = cv2.add(witch[:, :, 2], int(40*intensity))
            witch = apply_wavy_effect(witch, amplitude=4, frequency=30)
            img = cv2.addWeighted(witch, opacity, img_cv, 1-opacity, 0)
            result = Image.fromarray(img)
            result = add_sparkles(result, count=int(50*intensity))
            moon = Image.new("RGBA", result.size, (0,0,0,0))
            draw = ImageDraw.Draw(moon)
            moon_size = min(result.size)//4
            draw.ellipse((result.width-moon_size-20, 20, result.width-20, 20+moon_size), fill=(255,255,200,90))
            result = Image.alpha_composite(result.convert("RGBA"), moon).convert("RGB")
            result = ImageEnhance.Contrast(result).enhance(1.6*intensity)

        else:
            result = image

        st.image(result, caption=f"✨ {filter_choice} Look Applied", use_container_width=True)
        comment, rating = filter_comments.get(filter_choice, ("🌟 Original picture is lovely!", 5))
        st.markdown(f"**💬 Comment:** {comment}")
        st.markdown(f"**⭐ Rating:** {rating}/5")

        # -----------------------------
        # Fun floating particles (CSS)
        # -----------------------------
        if filter_choice != "None":
            icons = ["✨", "🪄", "🌙", "⭐", "🦇", "🎃", "👻", "🕷️", "🕸️"]
            html = ""
            for i in range(20):
                icon = random.choice(icons)
                x = random.randint(0, 100)
                size = random.randint(20, 60)
                duration = random.uniform(8, 25)
                delay = random.uniform(0, 10)
                rotation = random.randint(0, 360)
                html += (
                    f'<div style="position: fixed; left: {x}%; bottom: -10%; font-size: {size}px; '
                    f'transform: rotate({rotation}deg); animation: float{i} {duration}s linear {delay}s infinite;">{icon}</div>'
                    f'<style>@keyframes float{i} {{ 0% {{ transform: translateY(0) rotate({rotation}deg); opacity: 1; }} '
                    f'100% {{ transform: translateY(-120vh) rotate({rotation+360}deg); opacity: 0; }} }}</style>'
                )
            st.markdown(html, unsafe_allow_html=True)
