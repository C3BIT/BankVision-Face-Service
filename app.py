"""
OpenCV DNN Face Recognition Microservice
FastAPI service for face verification using OpenCV's SFace model

Uses OpenCV's built-in deep learning face recognition:
- YuNet face detector (fast, accurate)
- SFace recognizer (128-d embeddings, state-of-the-art accuracy)

Endpoints:
- POST /compare - Compare two face images
- GET /health - Health check
"""

import os
import io
import base64
import logging
import urllib.request
from typing import Optional, Tuple
import httpx
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition Service",
    description="OpenCV DNN-based face verification using SFace model",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODELS_DIR = "/app/models"
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Model URLs (OpenCV Zoo)
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Global model instances
face_detector = None
face_recognizer = None

def download_model(url: str, path: str):
    """Download model file if not exists"""
    if not os.path.exists(path):
        logger.info(f"Downloading model from {url}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        logger.info(f"Model downloaded to {path}")

def init_models():
    """Initialize face detector and recognizer"""
    global face_detector, face_recognizer

    # Download models if needed
    download_model(YUNET_URL, YUNET_MODEL)
    download_model(SFACE_URL, SFACE_MODEL)

    # Initialize YuNet face detector
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),  # Input size (will be updated per image)
        0.5,  # Score threshold (lowered for better detection)
        0.3,  # NMS threshold
        5000  # Top K
    )

    # Initialize SFace recognizer
    face_recognizer = cv2.FaceRecognizerSF.create(
        SFACE_MODEL,
        ""
    )

    logger.info("Face detection and recognition models initialized")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    init_models()

class CompareRequest(BaseModel):
    image1: str  # URL or base64
    image2: str  # URL or base64

class CompareResponse(BaseModel):
    matched: bool
    similarity: float
    confidence: float
    faces_detected: dict
    message: str

async def load_image_from_url(url: str) -> np.ndarray:
    """Load image from URL"""
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
    except Exception as e:
        logger.error(f"Failed to load image from URL: {url}, error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")

def load_image_from_base64(base64_str: str) -> np.ndarray:
    """Load image from base64 string"""
    try:
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        image_data = base64.b64decode(base64_str)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

async def load_image(image_input: str) -> np.ndarray:
    """Load image from URL or base64"""
    if image_input.startswith('http://') or image_input.startswith('https://'):
        return await load_image_from_url(image_input)
    else:
        return load_image_from_base64(image_input)

def detect_face(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect face and return aligned face for recognition
    Returns (face_bbox, aligned_face) or (None, None) if no face found
    """
    if image is None or face_detector is None:
        return None, None

    h, w = image.shape[:2]

    # Update detector input size to match image
    face_detector.setInputSize((w, h))

    # Detect faces
    _, faces = face_detector.detect(image)

    if faces is None or len(faces) == 0:
        return None, None

    # Get the largest face (by area)
    largest_face = max(faces, key=lambda f: f[2] * f[3])

    # Align face for recognition
    aligned_face = face_recognizer.alignCrop(image, largest_face)

    return largest_face, aligned_face

def get_face_embedding(aligned_face: np.ndarray) -> np.ndarray:
    """Extract 128-d face embedding using SFace"""
    if aligned_face is None or face_recognizer is None:
        return None

    # Get face feature (128-dimensional embedding)
    embedding = face_recognizer.feature(aligned_face)
    return embedding

def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float, bool]:
    """
    Compare two face embeddings
    Returns (similarity_score, confidence, is_match)
    """
    # Use cosine similarity (recommended for SFace)
    cosine_score = face_recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)

    # Also get L2 distance for reference
    l2_distance = face_recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_NORM_L2)

    # Thresholds (from OpenCV documentation)
    # Cosine: same person > 0.363, different person < 0.363
    # L2: same person < 1.128, different person > 1.128
    COSINE_THRESHOLD = 0.363
    L2_THRESHOLD = 1.128

    # Convert cosine score to percentage (0-100)
    # Cosine score ranges from -1 to 1, but typically 0.2-0.8 for faces
    # Map 0.363 (threshold) to 65%, 1.0 to 100%, 0.0 to 0%
    similarity = max(0, min(100, cosine_score * 100))

    # Determine match
    is_match = cosine_score >= COSINE_THRESHOLD

    # Calculate confidence based on how far from threshold
    if cosine_score >= COSINE_THRESHOLD:
        # Matched - confidence increases as we get further above threshold
        confidence = min(100, 70 + (cosine_score - COSINE_THRESHOLD) * 100)
    else:
        # Not matched - confidence in the "not match" decision
        confidence = min(100, 70 + (COSINE_THRESHOLD - cosine_score) * 100)

    logger.info(f"Cosine score: {cosine_score:.4f}, L2 distance: {l2_distance:.4f}, match: {is_match}")

    return similarity, confidence, is_match

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_ready = face_detector is not None and face_recognizer is not None
    return {
        "status": "healthy" if models_ready else "initializing",
        "service": "opencv-face-recognition",
        "version": "2.0.0",
        "model": "SFace (128-d embeddings)",
        "detector": "YuNet",
        "opencv_version": cv2.__version__,
        "models_ready": models_ready
    }

@app.post("/compare", response_model=CompareResponse)
async def compare_faces(request: CompareRequest):
    """
    Compare two face images using deep learning face embeddings

    - **image1**: URL or base64 encoded image
    - **image2**: URL or base64 encoded image

    Returns similarity score (0-100) and match result
    Uses OpenCV's SFace model for accurate face recognition
    """
    if face_detector is None or face_recognizer is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    logger.info("Received face comparison request")

    # Load images
    img1 = await load_image(request.image1)
    img2 = await load_image(request.image2)

    if img1 is None:
        raise HTTPException(status_code=400, detail="Failed to load image 1")
    if img2 is None:
        raise HTTPException(status_code=400, detail="Failed to load image 2")

    logger.info(f"Images loaded: img1={img1.shape}, img2={img2.shape}")

    # Detect and align faces
    face1_bbox, aligned_face1 = detect_face(img1)
    face2_bbox, aligned_face2 = detect_face(img2)

    faces_detected = {
        "image1": aligned_face1 is not None,
        "image2": aligned_face2 is not None
    }

    if aligned_face1 is None:
        logger.warning("No face detected in image 1")
        return CompareResponse(
            matched=False,
            similarity=0.0,
            confidence=0.0,
            faces_detected=faces_detected,
            message="No face detected in image 1"
        )

    if aligned_face2 is None:
        logger.warning("No face detected in image 2")
        return CompareResponse(
            matched=False,
            similarity=0.0,
            confidence=0.0,
            faces_detected=faces_detected,
            message="No face detected in image 2"
        )

    logger.info("Faces detected, extracting embeddings...")

    # Get face embeddings
    emb1 = get_face_embedding(aligned_face1)
    emb2 = get_face_embedding(aligned_face2)

    if emb1 is None or emb2 is None:
        return CompareResponse(
            matched=False,
            similarity=0.0,
            confidence=0.0,
            faces_detected=faces_detected,
            message="Failed to extract face features"
        )

    # Compare embeddings
    similarity, confidence, matched = compare_embeddings(emb1, emb2)

    logger.info(f"Comparison result: similarity={similarity:.2f}%, matched={matched}")

    return CompareResponse(
        matched=matched,
        similarity=round(similarity, 2),
        confidence=round(confidence, 2),
        faces_detected=faces_detected,
        message="Face comparison completed using SFace model"
    )

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "OpenCV Face Recognition Service",
        "version": "2.0.0",
        "models": {
            "detector": "YuNet (face_detection_yunet_2023mar)",
            "recognizer": "SFace (face_recognition_sface_2021dec)"
        },
        "features": [
            "128-dimensional face embeddings",
            "Cosine similarity matching",
            "Real-time performance",
            "Pure OpenCV implementation"
        ],
        "endpoints": {
            "compare": "POST /compare - Compare two face images",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5097))
    uvicorn.run(app, host="0.0.0.0", port=port)
