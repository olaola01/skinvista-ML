import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
base_dir = "dataset"
categories = ["acne", "eczema", "keloids", "fungal_infections", "pseudofolliculitis_barbae",
              "ringworm", "vitiligo"]
IMG_SIZE = 224
PLOTS_DIR = "plots"
CONFUSING_IMAGES_DIR = "confusing_images"
BATCH_SIZE = 16
TARGET_SAMPLES = 960
TEST_SAMPLES_LIMIT = 200

# Create directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CONFUSING_IMAGES_DIR, exist_ok=True)

# Check GPU availability
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")


# Data augmentation
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image


# Data generator
def data_generator(image_paths, labels, class_weight_dict, batch_size, augment=True):
    while True:
        indices = np.random.permutation(len(image_paths))
        for start in range(0, len(image_paths), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_images = []
            batch_labels = []
            batch_weights = []

            for i in batch_indices:
                img = cv2.imread(image_paths[i])
                if img is None:
                    logger.warning(f"Could not load image {image_paths[i]}")
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                if augment:
                    img = augment_image(img)
                batch_images.append(img)
                batch_labels.append(labels[i])
                batch_weights.append(class_weight_dict[labels[i]])

            if not batch_images:
                continue

            batch_images = tf.convert_to_tensor(np.array(batch_images), dtype=tf.float32)
            batch_labels = tf.convert_to_tensor(np.array(batch_labels), dtype=tf.int32)
            batch_weights = tf.convert_to_tensor(np.array(batch_weights), dtype=tf.float32)

            yield batch_images, batch_labels, batch_weights


# Load and preprocess data
def load_data():
    images = []
    labels = []
    image_paths = []
    logger.info("Loading dataset...")
    for label, category in enumerate(categories):
        path = os.path.join(base_dir, category)
        if not os.path.exists(path):
            logger.warning(f"Directory {path} not found. Skipping.")
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load {img_path}")
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                images.append(img)
                labels.append(label)
                image_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error loading {img_path}: {e}")
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    if len(images) == 0:
        raise ValueError("No images loaded. Check dataset folder structure.")
    logger.info(f"Loaded {len(images)} images with labels: {Counter(labels)}")
    return images, labels, image_paths


# Oversample minority classes
def oversample_minority_classes(X, y, image_paths, min_samples=TARGET_SAMPLES, test_mode=False):
    label_counts = Counter(y)
    logger.info(f"Class distribution: {{categories[k]: v for k, v in label_counts.items()}}")
    X_balanced = []
    y_balanced = []
    paths_balanced = []
    for label in label_counts:
        X_class = X[y == label]
        y_class = y[y == label]
        paths_class = [image_paths[i] for i in range(len(y)) if y[i] == label]
        target_samples = min_samples if not test_mode else min(TARGET_SAMPLES, max(TEST_SAMPLES_LIMIT, len(X_class)))
        if len(X_class) < target_samples:
            logger.info(f"Oversampling class {categories[label]} from {len(X_class)} to {target_samples}")
            X_class_resampled, y_class_resampled, paths_resampled = resample(
                X_class, y_class, paths_class, replace=True, n_samples=target_samples, random_state=42
            )
        else:
            X_class_resampled, y_class_resampled, paths_resampled = X_class, y_class, paths_class
        X_balanced.append(X_class_resampled)
        y_balanced.append(y_class_resampled)
        paths_balanced.extend(paths_resampled)
    return np.vstack(X_balanced), np.hstack(y_balanced), paths_balanced


# Load data
try:
    X, y, image_paths = load_data()
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    exit(1)

# Split into train+val and test sets
X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
    X, y, image_paths, test_size=0.1, random_state=42, stratify=y
)
# Oversample train+val
X_balanced, y_balanced, paths_balanced = oversample_minority_classes(X_temp, y_temp, paths_temp)
# Create test set with limited oversampling
X_test_balanced, y_test_balanced, paths_test_balanced = oversample_minority_classes(
    X_test, y_test, paths_test, test_mode=True
)
# Split train+val into train and val
X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
    X_balanced, y_balanced, paths_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
logger.info(f"Training samples: {X_train.shape}")
logger.info(f"Validation samples: {X_val.shape}")
logger.info(f"Test samples: {X_test_balanced.shape}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
logger.info(f"Class weights: {class_weight_dict}")

# Free memory
del X, y, X_temp, y_temp, X_balanced, y_balanced
import gc

gc.collect()

# Build model
try:
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(categories), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0002), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    logger.info("Model compiled successfully")
except Exception as e:
    logger.error(f"Failed to build model: {e}")
    exit(1)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train model
try:
    logger.info("Starting training...")
    steps_per_epoch = len(paths_train) // BATCH_SIZE
    validation_steps = len(paths_val) // BATCH_SIZE
    history = model.fit(
        data_generator(paths_train, y_train, class_weight_dict, BATCH_SIZE, augment=True),
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=data_generator(paths_val, y_val, class_weight_dict, BATCH_SIZE, augment=False),
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    logger.info(f"Training completed. Final training accuracy: {history.history['accuracy'][-1]:.4f}, "
                f"validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
except Exception as e:
    logger.error(f"Training failed: {e}")
    exit(1)

# Plot metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_plot.png'))
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'loss_plot.png'))
plt.close()

# Evaluate on validation set
y_pred = model.predict(X_val, batch_size=BATCH_SIZE)
y_pred_classes = np.argmax(y_pred, axis=1)
logger.info("\nValidation Classification Report:")
logger.info(classification_report(y_val, y_pred_classes, target_names=categories))
kappa = cohen_kappa_score(y_val, y_pred_classes)
logger.info(f"Validation Cohen's Kappa: {kappa:.4f}")

# Confusion matrix for validation
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_val.png'))
plt.close()

# Evaluate on test set
y_test_pred = model.predict(X_test_balanced, batch_size=BATCH_SIZE)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
logger.info("\nTest Classification Report:")
logger.info(classification_report(y_test_balanced, y_test_pred_classes, target_names=categories))
test_kappa = cohen_kappa_score(y_test_balanced, y_test_pred_classes)
logger.info(f"Test Cohen's Kappa: {test_kappa:.4f}")

# Confusion matrix for test
cm_test = confusion_matrix(y_test_balanced, y_test_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_test.png'))
plt.close()

# Identify confusing images (validation)
logger.info("Identifying confusing images...")
confusing_log = []
for i, (true_label, pred_label, img_path, pred_prob) in enumerate(zip(y_val, y_pred_classes, paths_val, y_pred)):
    if true_label != pred_label:
        confidence = pred_prob[pred_label]
        true_category = categories[true_label]
        pred_category = categories[pred_label]
        confusing_subdir = os.path.join(CONFUSING_IMAGES_DIR, true_category)
        os.makedirs(confusing_subdir, exist_ok=True)
        new_path = os.path.join(confusing_subdir, os.path.basename(img_path))
        try:
            shutil.copy(img_path, new_path)
            confusing_log.append({
                'original_path': img_path,
                'new_path': new_path,
                'true_label': true_category,
                'predicted_label': pred_category,
                'confidence': float(confidence)
            })
            logger.info(
                f"Copied {img_path} to {new_path} (True: {true_category}, Predicted: {pred_category}, Confidence: {confidence:.2f})")
        except Exception as e:
            logger.error(f"Error copying {img_path}: {e}")
with open(os.path.join(CONFUSING_IMAGES_DIR, 'confusing_images_log.txt'), 'w') as f:
    for entry in confusing_log:
        f.write(
            f"Original: {entry['original_path']}, New: {entry['new_path']}, True: {entry['true_label']}, Predicted: {entry['predicted_label']}, Confidence: {entry['confidence']:.2f}\n")

# Save model
model.save("skin_vista_model-01.keras")
logger.info("Model saved successfully")

# Test on training images
logger.info("Testing on first 10 training images:")
for i in range(min(10, len(X_train))):
    img = X_train[i:i + 1]
    pred = model.predict(img, batch_size=1)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    logger.info(f"Image {i}: Predicted {categories[class_idx]} ({confidence:.2f}), True: {categories[y_train[i]]}")

# Visualize Image 4
plt.figure()
plt.imshow(X_train[4])
plt.title(f"True Label: {categories[y_train[4]]}")
plt.axis('off')
plt.savefig(os.path.join(PLOTS_DIR, 'sample_image_4.png'))
plt.close()


# Predict on a single image
def predict_image(image_path, confidence_threshold=0.5):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image {image_path}")
            return None, None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, batch_size=1)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        logger.info(f"Raw probabilities for {image_path}: {prediction[0]}")
        if confidence < confidence_threshold:
            return "no_diagnosis", confidence
        return categories[class_idx], confidence
    except Exception as e:
        logger.error(f"Error predicting {image_path}: {e}")
        return None, None


# Test prediction
test_image_path = "dataset/eczema/desquamation-1.jpg"
if os.path.exists(test_image_path):
    condition, confidence = predict_image(test_image_path)
    logger.info(f"Test Prediction: {condition} ({confidence * 100:.0f}% confidence)")
else:
    logger.error(f"Test image {test_image_path} not found.")
