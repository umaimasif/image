from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from PIL import Image,ImageDraw, ImageFont, ImageFilter
import io
import cv2
import numpy as np

app = FastAPI()
@app.post("/grayscale")
async def convert_to_grayscale(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    gray_img = img.convert("L")
    
    output_buffer = io.BytesIO()
    gray_img.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="image/png")

@app.post("/resize")
async def resize_image(
    width: int = Query(..., description="New width of the image"),
    height: int = Query(..., description="New height of the image"),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    resized = img.resize((width, height))
    buffer = io.BytesIO()
    resized.save(buffer, format=img.format)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type=f"image/{img.format.lower()}")

@app.post("/watermark")
async def add_watermark(
    text: str = Query(..., description="Watermark text"),
    position: str = Query("bottom-right", description="Position: top-left, top-right, bottom-left, bottom-right"),
    file: UploadFile = File(...)
):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")  # Ensure RGBA for transparency

    # Create watermark layer
    txt_layer = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt_layer)

    # Default font
    font_size = max(20, img.size[0] // 20)
    font = ImageFont.load_default()

    # Calculate text size using textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 10
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = img.width - text_width - margin, margin
    elif position == "bottom-left":
        x, y = margin, img.height - text_height - margin
    else:  # bottom-right
        x, y = img.width - text_width - margin, img.height - text_height - margin

    # Draw text
    draw.text((x, y), text, fill=(255,255,255,128), font=font)  # semi-transparent white

    # Combine original image with watermark
    watermarked = Image.alpha_composite(img, txt_layer)

    # Save to buffer
    buffer = io.BytesIO()
    watermarked = watermarked.convert("RGB")  # remove alpha for JPG
    buffer_format = "JPEG" if file.filename.lower().endswith((".jpg", ".jpeg")) else "PNG"
    watermarked.save(buffer, format=buffer_format)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type=f"image/{buffer_format.lower()}")

@app.post("/logo-watermark")
async def add_logo_watermark(
    position: str = Query("bottom-right", description="Position: top-left, top-right, bottom-left, bottom-right"),
    file: UploadFile = File(..., description="Main image"),
    logo_file: UploadFile = File(..., description="Logo image (PNG recommended with transparency)")
):
    # Read main image
    main_bytes = await file.read()
    main_img = Image.open(io.BytesIO(main_bytes)).convert("RGBA")

    # Read logo image
    logo_bytes = await logo_file.read()
    logo_img = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")

    # Resize logo if bigger than main image
    max_width = main_img.width // 5
    max_height = main_img.height // 5
    logo_ratio = min(max_width / logo_img.width, max_height / logo_img.height, 1)
    new_size = (int(logo_img.width * logo_ratio), int(logo_img.height * logo_ratio))
    logo_img = logo_img.resize(new_size, Image.Resampling.LANCZOS)  # fixed

    # Calculate position
    margin = 10
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = main_img.width - logo_img.width - margin, margin
    elif position == "bottom-left":
        x, y = margin, main_img.height - logo_img.height - margin
    else:  # bottom-right
        x, y = main_img.width - logo_img.width - margin, main_img.height - logo_img.height - margin

    # Paste logo
    main_img.paste(logo_img, (x, y), logo_img)  # use logo_img as mask for transparency

    # Save to buffer
    buffer = io.BytesIO()
    output_format = "JPEG" if file.filename.lower().endswith((".jpg", ".jpeg")) else "PNG"
    main_img = main_img.convert("RGB") if output_format == "JPEG" else main_img
    main_img.save(buffer, format=output_format)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type=f"image/{output_format.lower()}")

@app.post("/filter")
async def apply_filter(
    filter_type: str = Query(..., description="Filter type: blur, sharpen, edge"),
    file: UploadFile = File(...)
):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply filter
    if filter_type.lower() == "blur":
        filtered_img = img.filter(ImageFilter.BLUR)
    elif filter_type.lower() == "sharpen":
        filtered_img = img.filter(ImageFilter.SHARPEN)
    elif filter_type.lower() == "edge":
        filtered_img = img.filter(ImageFilter.FIND_EDGES)
    else:
        return {"error": "Invalid filter type. Choose: blur, sharpen, edge"}

    # Save to buffer
    buffer = io.BytesIO()
    output_format = "JPEG" if file.filename.lower().endswith((".jpg", ".jpeg")) else "PNG"
    filtered_img.save(buffer, format=output_format)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type=f"image/{output_format.lower()}")

@app.post("/canny-edge")
async def canny_edge(file: UploadFile = File(...), threshold1: int = 100, threshold2: int = 200):
    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Convert edges to RGB for saving
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Encode as PNG
    _, buffer = cv2.imencode(".png", edges_rgb)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.post("/face-detect")
async def detect_faces(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Encode as PNG
    _, buffer = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Encode as PNG
    _, buffer = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


