from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load model
try:
    model = load_model("skin_vista_model-01.keras")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise Exception("Model loading failed")

# Categories
categories = ["acne", "eczema", "keloids", "fungal_infections", "pseudofolliculitis_barbae",
              "ringworm", "vitiligo"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        # Read the uploaded image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocess image
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = preprocess_input(image * 255.0)  # Apply MobileNetV2 preprocessing (scales to [-1, 1])
        image_array = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image_array)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        condition = categories[class_idx]

        logger.info(f"Prediction: {condition} with confidence {confidence:.4f}")

        # Return result
        return {"condition": condition, "confidence": confidence}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
