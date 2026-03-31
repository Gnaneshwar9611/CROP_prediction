"""
AgriScan Backend — Crop Disease Prediction API
Runs fully offline using local .h5 TensorFlow/Keras models.
Returns human-readable disease names instead of numeric class indices.
"""

import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np

# ── Keras compatibility patch ────────────────────────────────────
# Some .h5 models saved with newer Keras include 'quantization_config'
# in layer configs, which older Keras versions don't recognise.
# This patch silently strips that key so load_model succeeds.
import keras

_PATCH_LAYERS = [
    keras.layers.Dense,
    keras.layers.Conv2D,
    keras.layers.DepthwiseConv2D,
    keras.layers.BatchNormalization,
]

for _layer_cls in _PATCH_LAYERS:
    _orig = _layer_cls.from_config

    @classmethod  # type: ignore[misc]
    def _patched_from_config(cls, config, _orig=_orig):
        config.pop("quantization_config", None)
        return _orig.__func__(cls, config)

    _layer_cls.from_config = _patched_from_config

import tensorflow as tf  # import after patch

from utils import preprocess_image

# ── Model configuration ─────────────────────────────────────────
# Directory that holds the four .h5 files
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Per-crop settings: filename and the image size the model expects
CROP_CONFIG = {
    "chilli": {
        "file": "chilli_model.h5",
        "input_size": (128, 128),
    },
    "rice": {
        "file": "rice_model.h5",
        "input_size": (224, 224),
    },
    "finger_millet": {
        "file": "finger_millet_model.h5",
        "input_size": (224, 224),
    },
    "sugarcane": {
        "file": "sugarcane_model.h5",
        "input_size": (224, 224),
    },
}

# ── Disease class names ──────────────────────────────────────────
# IMPORTANT: The order MUST match the alphabetical folder order used
# during training (Keras image_dataset_from_directory sorts folders
# alphabetically by default).
#
# Model output sizes:
#   chilli         → 8 classes
#   rice           → 6 classes
#   finger_millet  → 6 classes
#   sugarcane      → 3 classes

CLASS_NAMES: dict[str, list[str]] = {
    "chilli": [
        "Bacterial Spot",
        "Cercospora Leaf Spot",
        "Healthy",
        "Leaf Curl",
        "Leaf Spot",
        "Powdery Mildew",
        "Whitefly",
        "Yellowish",
    ],  # 8 classes
    "rice": [
        "Bacterial Leaf Blight",
        "Blast",
        "Brown Spot",
        "Healthy",
        "Leaf Scald",
        "Tungro",
    ],  # 6 classes
    "finger_millet": [
        "Blast",
        "Healthy",
        "Leaf Spot",
        "Millet Rust",
        "Mosaic Streak Virus",
        "Wilt",
    ],  # 6 classes
    "sugarcane": [
        "Healthy",
        "Red Rot",
        "Rust",
    ],  # 3 classes
}

# ── Load all models once at startup ─────────────────────────────
models: dict[str, tf.keras.Model | None] = {}

for crop_key, cfg in CROP_CONFIG.items():
    model_path = os.path.join(MODELS_DIR, cfg["file"])
    try:
        models[crop_key] = tf.keras.models.load_model(model_path, compile=False)
        print(f"✔ Loaded model for '{crop_key}' from {cfg['file']}")
    except Exception as exc:
        models[crop_key] = None
        print(f"✘ Failed to load model for '{crop_key}': {exc}")

# ── FastAPI application ─────────────────────────────────────────
app = FastAPI(
    title="AgriScan API",
    description="Crop disease prediction — fully offline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Health-check / welcome endpoint."""
    return {"message": "Welcome to the AgriScan API"}


@app.post("/predict")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    """
    Accept a crop name and an image, return predicted disease name
    and confidence score as a percentage.
    """

    # 1. Validate crop name
    if crop not in CROP_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop '{crop}'. Choose from: {list(CROP_CONFIG)}",
        )

    # 2. Check that model loaded successfully
    model = models[crop]
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model for '{crop}' is not available (failed to load at startup).",
        )

    # 3. Read and preprocess the uploaded image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        target_size = CROP_CONFIG[crop]["input_size"]
        processed = preprocess_image(image, target_size=target_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image processing error: {exc}")

    # 4. Run prediction
    try:
        predictions = model.predict(processed)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions, axis=1)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # 5. Convert class index → human-readable disease name
    disease_name = CLASS_NAMES[crop][predicted_class]

    # 6. Convert confidence to percentage (rounded to 2 decimal places)
    confidence_pct = round(confidence * 100, 2)

    return {
        "crop": crop,
        "disease": disease_name,
        "confidence": confidence_pct,
    }