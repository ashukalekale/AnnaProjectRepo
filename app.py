from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load disease model
try:
    disease_model = load_model("crop_disease_model.h5")
    print("Disease model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    disease_model = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Disease classes mapping
disease_classes = {
    0: ("Mango Anthracnose", "Apply copper fungicide, remove infected fruits."),
    1: ("Mango Bacterial Canker", "Prune affected branches, apply bactericide."),
    2: ("Mango Cutting Weevil", "Remove infested fruits, spray insecticide."),
    3: ("Mango Die Back", "Cut diseased limbs and apply fungicide."),
    4: ("Mango Gall Midge", "Use pheromone traps and neem spray."),
    5: ("Mango Healthy", "No disease detected."),
    6: ("Mango Powdery Mildew", "Apply sulfur fungicide."),
    7: ("Mango Sooty Mould", "Wash leaves, control honeydew insects."),
    8: ("Cotton Bacterial Blight", "Use resistant varieties and copper spray."),
    9: ("Cotton Curl Virus", "Use virus-free seeds, control whiteflies."),
    10: ("Cotton Fusarium Wilt", "Rotate crops, avoid overwatering."),
    11: ("Cotton Healthy", "No disease detected."),
    12: ("Watermelon Anthracnose", "Spray fungicide, avoid overhead irrigation."),
    13: ("Watermelon Downy Mildew", "Use resistant varieties, apply fungicide."),
    14: ("Watermelon Healthy", "No disease detected."),
    15: ("Watermelon Mosaic Virus", "Remove infected plants, control aphids.")
}

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess uploaded image
def preprocess_image(file):
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, 224, 224, 3)
    except Exception as e:
        print("Error processing image:", e)
        return None

@app.route("/")
def index():
    return render_template("disease.html", result=None, solution=None)

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    result = None
    solution = None

    symptoms = request.form.get("symptoms")
    image = request.files.get("image")

    # If symptom text is provided — simple rule-based detection
    if symptoms:
        symptoms_lower = symptoms.lower()

        if "yellow" in symptoms_lower:
            result = "Possible Disease: Leaf Yellowing"
            solution = "Add nitrogen fertilizer and check irrigation."
        elif "spot" in symptoms_lower:
            result = "Possible Disease: Leaf Spot"
            solution = "Use fungicide and remove affected leaves."
        else:
            result = "Disease not identified"
            solution = "Consult an expert or upload an image."
        return render_template("disease.html", result=result, solution=solution)

    # If image is uploaded
    if image and allowed_file(image.filename):
        if disease_model is None:
            return render_template("disease.html", result="Model not loaded", solution=None)

        img_array = preprocess_image(image)
        if img_array is None:
            return render_template("disease.html", result="Error processing image", solution=None)

        pred = disease_model.predict(img_array)
        class_id = np.argmax(pred)

        disease_name, remedy = disease_classes.get(class_id, ("Unknown Disease", "No remedy found."))

        result = f"Predicted Disease: {disease_name}"
        solution = f"Suggested Remedy: {remedy}"

        return render_template("disease.html", result=result, solution=solution)

    return render_template("disease.html", result="No file or symptoms provided", solution=None)

if __name__ == "__main__":
    app.run(debug=True)
