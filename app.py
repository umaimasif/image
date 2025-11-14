import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import io

st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("Image Processing App")

# Upload main image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image", use_column_width=True)

    # Select action
    action = st.selectbox(
        "Choose an action",
        [
            "Grayscale",
            "Resize",
            "Text Watermark",
            "Logo Watermark",
            "Filter",
            "Canny Edge Detection",
            "Face Detection",
            "Remove Background"
        ]
    )

    # Extra inputs based on action
    width, height = None, None
    text, position = None, None
    logo_file = None
    filter_type = None
    threshold1, threshold2 = 100, 200

    if action == "Resize":
        width = st.number_input("Width", min_value=1, value=img.width)
        height = st.number_input("Height", min_value=1, value=img.height)
    elif action == "Text Watermark":
        text = st.text_input("Watermark text", "Sample")
        position = st.selectbox("Position", ["top-left", "top-right", "bottom-left", "bottom-right"])
    elif action == "Logo Watermark":
        logo_file = st.file_uploader("Upload logo", type=["png", "jpg", "jpeg"])
        position = st.selectbox("Position", ["top-left", "top-right", "bottom-left", "bottom-right"])
    elif action == "Filter":
        filter_type = st.selectbox("Filter type", ["blur", "sharpen", "edge"])
    elif action == "Canny Edge Detection":
        threshold1 = st.slider("Threshold1", 0, 500, 100)
        threshold2 = st.slider("Threshold2", 0, 500, 200)

    # Apply action
    if st.button("Apply"):
        result = img.copy()

        if action == "Grayscale":
            result = result.convert("L")
        elif action == "Resize":
            result = result.resize((int(width), int(height)))
        elif action == "Text Watermark" and text:
            result = result.convert("RGBA")
            txt_layer = Image.new("RGBA", result.size, (255,255,255,0))
            draw = ImageDraw.Draw(txt_layer)
            font_size = max(20, result.size[0] // 20)
            font = ImageFont.load_default()
            bbox = draw.textbbox((0,0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            margin = 10
            if position == "top-left":
                x, y = margin, margin
            elif position == "top-right":
                x, y = result.width - text_width - margin, margin
            elif position == "bottom-left":
                x, y = margin, result.height - text_height - margin
            else:
                x, y = result.width - text_width - margin, result.height - text_height - margin
            draw.text((x, y), text, fill=(255,255,255,128), font=font)
            result = Image.alpha_composite(result, txt_layer).convert("RGB")
        elif action == "Logo Watermark" and logo_file:
            result = result.convert("RGBA")
            logo = Image.open(logo_file).convert("RGBA")
            max_width = result.width // 5
            max_height = result.height // 5
            logo_ratio = min(max_width / logo.width, max_height / logo.height, 1)
            new_size = (int(logo.width * logo_ratio), int(logo.height * logo_ratio))
            logo = logo.resize(new_size, Image.Resampling.LANCZOS)
            margin = 10
            if position == "top-left":
                x, y = margin, margin
            elif position == "top-right":
                x, y = result.width - logo.width - margin, margin
            elif position == "bottom-left":
                x, y = margin, result.height - logo.height - margin
            else:
                x, y = result.width - logo.width - margin, result.height - logo.height - margin
            result.paste(logo, (x, y), logo)
            result = result.convert("RGB")
        elif action == "Filter":
            if filter_type == "blur":
                result = result.filter(ImageFilter.BLUR)
            elif filter_type == "sharpen":
                result = result.filter(ImageFilter.SHARPEN)
            elif filter_type == "edge":
                result = result.filter(ImageFilter.FIND_EDGES)
        elif action == "Canny Edge Detection":
            img_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            result = Image.fromarray(edges)
        elif action == "Face Detection":
            img_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0,255,0), 2)
            result = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        elif action == "Remove Background":
            img_array = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            result_array = cv2.bitwise_and(img_array, img_array, mask=mask)
            result = Image.fromarray(cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB))

        st.image(result, caption="Processed Image", use_column_width=True)
        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), file_name="processed.png", mime="image/png")
