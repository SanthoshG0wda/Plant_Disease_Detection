import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model & classes
model = tf.keras.models.load_model("new_model.keras")

with open("crop_class_names.json") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print(img_array)

    preds = model.predict(img_array)[0]

    confidence = np.max(preds) * 100
    index = np.argmax(preds)
    label = class_names[index]

    # Entropy check
    entropy = -np.sum(preds * np.log(preds + 1e-10))

    if label == "Unknown" or confidence < 70 or entropy > 1.2:
        return "Unknown Crop", confidence

    return label, confidence


crop, conf = predict_image("0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG")
print(f"Prediction: {crop} ({conf:.2f}%)")
