import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# Define categories and paths
categories = ["acne", "eczema", "keloids", "fungal_infections", "pseudofolliculitis_barbae", "ringworm", "vitiligo"]
sample_folder = "dataset-sample"  # Replace with your sample folder path
model_path = "skin_vista_model-01.keras"  # Replace with your model path

# Load the model
model = tf.keras.models.load_model(model_path)

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize to 224x224
    img = np.array(img) / 255.0  # Normalize to [0,1]
    if img.shape[-1] == 4:  # Remove alpha channel if present
        img = img[:, :, :3]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Initialize lists for the table
image_ids = []
model_predictions = []
model_confidences = []
actual_predictions = []

# Process each image
for category in categories:
    category_path = os.path.join(sample_folder, category)
    for image_name in os.listdir(category_path):
        if image_name.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(category_path, image_name)
            # Use original filename as Image ID
            image_id = image_name
            # Preprocess and predict
            img = preprocess_image(image_path)
            prediction = model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100  # Convert to percentage
            # Append to lists
            image_ids.append(image_id)
            model_predictions.append(categories[predicted_class])
            model_confidences.append(f"{confidence:.0f}%")
            actual_predictions.append(category)
            # Print verification with ✅ or ❌
            match = "✅" if categories[predicted_class] == category else "❌"
            print(f"Image: {image_name}, Model Prediction: {categories[predicted_class]}, "
                  f"Actual Prediction: {category}, Confidence: {confidence:.0f}% {match}")

# Create DataFrame for the table
validation_table = pd.DataFrame({
    "Image ID": image_ids,
    "Model Prediction": model_predictions,
    "Actual Prediction": actual_predictions,
    "Model Confidence": model_confidences,
    "Expert Diagnosis": [""] * len(image_ids),
    "Likert Rating (1–5)": [""] * len(image_ids),
    "Comments": [""] * len(image_ids)
})

# Save to CSV for reference
validation_table.to_csv("validation_table.csv", index=False)
print("\nValidation table saved to validation_table.csv")
print(validation_table.head())