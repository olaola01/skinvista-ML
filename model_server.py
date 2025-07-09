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
    print("Model loaded successfully")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    print("Model loading failed")
    raise Exception("Model loading failed")

# Categories
categories = ["acne", "eczema", "keloids", "fungal_infections", "pseudofolliculitis_barbae",
              "ringworm", "vitiligo"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize and preprocess
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)  # Keep in [0, 255] for preprocess_input
        image = preprocess_input(image)  # MobileNetV2 preprocessing
        image_array = np.expand_dims(image, axis=0)

        # Verify input shape
        logger.info(f"Input shape: {image_array.shape}")

        # Make prediction
        prediction = model.predict(image_array)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        condition = categories[class_idx]

        logger.info(f"Prediction: {condition} with confidence {confidence:.4f}")
        return {"condition": condition, "confidence": confidence}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
