from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import boto3
import cv2
import numpy as np
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Rekognition setup with environment fallback
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Create Rekognition client with error handling
def get_rekognition_client():
    try:
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            return boto3.client(
                'rekognition',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
        else:
            return boto3.client('rekognition', region_name=AWS_REGION)
    except Exception as e:
        print(f"Error initializing Rekognition client: {e}")
        raise

collection_id = "new_face_collection"
image_to_name = {
    "imag1.jpg": "Akshat",
    "photo.jpg": "Akshat"
}
safety_keywords = ['helmet', 'hardhat', 'safety vest', 'vest', 'goggles', 'boots', 'gloves']

@app.post("/analyze/")
async def analyze_frame(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        return {"error": "Uploaded file is not an image."}
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Could not decode image. Please upload a valid image file."}
        _, buffer = cv2.imencode('.jpg', image)
        img_bytes = buffer.tobytes()
    except Exception as e:
        return {"error": f"Image processing error: {str(e)}"}

    rekognition = get_rekognition_client()
    objects = []
    faces = []

    # Object detection
    try:
        obj_response = rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=75
        )
        for label in obj_response.get("Labels", []):
            name = label.get("Name","")
            confidence = label.get("Confidence",0)
            if any(keyword in name.lower() for keyword in safety_keywords):
                objects.append({"name": name, "confidence": confidence})
    except Exception as e:
        return {"error": f"Object detection error: {str(e)}"}

    # Face recognition
    try:
        face_response = rekognition.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': img_bytes},
            FaceMatchThreshold=95,
            MaxFaces=5
        )
        for match in face_response.get("FaceMatches", []):
            face = match.get("Face", {})
            face_id = face.get("ExternalImageId", "")
            similarity = match.get("Similarity", 0)
            name = image_to_name.get(face_id, "Person")
            faces.append({"name": name, "similarity": similarity})
    except Exception as e:
        return {"error": f"Face recognition error: {str(e)}"}

    return {
        "detected_objects": objects,
        "recognized_faces": faces
    }
