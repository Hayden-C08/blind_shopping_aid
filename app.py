from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from flask_cors import CORS

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from paddleocr import PaddleOCR
from gtts import gTTS
import google.generativeai as genai
import gdown

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/tmp/photo_uploads'  # use /tmp on Render (temp storage)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# === Model download from Google Drive ===
MODEL_PATH = 'resnet_model_new2.keras'
MODEL_DRIVE_ID = '1UgkW26TpNVRJV-9djaFbCayzYLDnQ0CM'  # <<< Replace with your actual file ID

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}', MODEL_PATH, quiet=False)

# === ML Setup ===
class_names = ['apple', 'banana', 'carrot', 'cucumber', 'mango', 'onion', 'orange', 'packaged_product', 'potato']
model = tf.keras.models.load_model(MODEL_PATH)
ocr = PaddleOCR(use_angle_cls=True, lang="en", rec_algorithm="CRNN", rec_image_shape="3, 32, 320", det_db_box_thresh=0.3, use_dilation=True)
genai.configure(api_key="AIzaSyD6yyxXnumQZsvI6BBh1DdlNUdx2buYH5s")
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(contrast, -1, kernel)
    return sharp

@app.route('/upload', methods=['POST'])
def upload_and_process():
    if 'file' not in request.files:
        abort(400, description="No file part")
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No selected file")
    if not allowed_file(file.filename):
        abort(415, description="File type not allowed")
    if request.content_length > MAX_FILE_SIZE:
        abort(413, description="File too large (max 10MB)")

    # Save file to /tmp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = secure_filename(file.filename)
    filename = f"{timestamp}_{original_filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Load & Predict with ResNet
    img = image.load_img(save_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = class_names[predicted_class]

    # Generate classification audio
    tts_text = f"You are holding a {class_name}"
    tts = gTTS(text=tts_text, lang='en')
    tts_path = os.path.join(UPLOAD_FOLDER, "class_output.mp3")
    tts.save(tts_path)

    output_info = {
        "class": class_name,
        "confidence": float(confidence),
        "class_audio": f"/audio/class_output.mp3"
    }

    if class_name == 'packaged_product':
        image_cv = cv2.imread(save_path)
        processed = preprocess_image_for_ocr(image_cv)
        processed_path = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")
        cv2.imwrite(processed_path, processed)

        results = ocr.ocr(processed_path, cls=True)
        extracted_text = "\n".join([word_info[1][0] for line in results for word_info in line])

        if not extracted_text.strip():
            output_info["ocr_text"] = "No text detected"
        else:
            prompt = f"""
            You are an intelligent assistant that extracts structured information from unformatted text.
            Analyze the given text and extract the following details:
            - Product Name
            - Flavor (if mentioned)
            - Price(just mention number no currency)
            - Manufacturing Date (MFG)
            - Expiry Date (EXP calculate if best before found but date not mentioned)
            - If expired(try to use the best before statement if expiry date not mentioned)

            please use ur prior database knowledge of all brand names and prices before determining an output.
            If any detail is missing, return 'Not Found' for that field.
            the date should always be in the format "the DD of month name YYYY" and dont give me any other output other than the ones ive asked and in that rxact format no extra info
            the DD should have 'st' or 'rd' or 'th' according to the context
            Here is the extracted text from the product:
            {extracted_text}

            Return in this format:
                - Product Name: <name or 'Not Found'>
                - Flavor: <flavor or 'Not Found'>
                - Price: <amount or 'Not Found'>
                - Manufacturing Date: <date or 'Not Found'>
                - Expiry Date: <date or 'Not Found'>
                - Product expiry: <yesproduct has expired or no product hasnt expired or cannot be determined>
            """
            response = gemini_model.generate_content(prompt)
            output_text = response.text.strip()

            tts = gTTS(text=output_text, lang='en')
            ocr_audio_path = os.path.join(UPLOAD_FOLDER, "ocr_output.mp3")
            tts.save(ocr_audio_path)

            output_info.update({
                "ocr_text": extracted_text,
                "gemini_output": output_text,
                "ocr_audio": f"/audio/ocr_output.mp3"
            })

    return jsonify(output_info), 200

@app.route('/audio/<filename>')
def serve_audio(filename):
    return app.send_from_directory(UPLOAD_FOLDER, filename)

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
