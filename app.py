from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "model/clf_digits.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
model = joblib.load(MODEL_PATH)

# Preprocessing dari file gambar
def preprocess_image(img):
    img = img.convert("L")  # ubah ke grayscale
    img = img.resize((8, 8), Image.LANCZOS)  # Resize dengan metode baru
    arr = np.array(img)
    arr = 16.0 * arr / 255.0  # Skala 0–16
    flat = arr.flatten().reshape(1, -1)
    return flat

# -------------------- ENDPOINT 1 --------------------
# Kirim array pixels langsung (misalnya dari frontend numerik)
@app.route("/predict-pixel", methods=["POST"])
def predict_pixel():
    data = request.get_json()
    if not data or "pixels" not in data:
        return jsonify({"error": "Harap kirim JSON dengan key 'pixels'."}), 400

    try:
        pixels = np.array(data["pixels"]).reshape(1, -1)
        prediction = int(model.predict(pixels)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- ENDPOINT 2 --------------------
# Kirim file gambar (.png, .jpg)
@app.route("/predict-file", methods=["POST"])
def predict_file():
    if 'image' not in request.files:
        return jsonify({"error": "Harap kirim file dengan key 'image'."}), 400

    try:
        file = request.files['image']
        img = Image.open(file.stream)
        pixels = preprocess_image(img)
        prediction = int(model.predict(pixels)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint test
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API prediksi digit aktif. Gunakan /predict-pixel atau /predict-file."})

if __name__ == "__main__":
    app.run(debug=True)
