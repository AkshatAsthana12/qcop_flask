from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import boto3
import cv2
import numpy as np

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HARDCODED AWS CREDENTIALS (use only for testing!) ---
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "AKIAUH4GU4K2X5WAPORN"
AWS_SECRET_ACCESS_KEY = "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"
# ---------------------------------------------------------

def get_rekognition_client():
    try:
        return boto3.client(
            'rekognition',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
    except Exception as e:
        print(f"Error initializing Rekognition client: {e}")
        raise

safety_keywords = ['helmet', 'hardhat', 'safety vest', 'vest', 'goggles', 'boots', 'gloves']

@app.post("/analyze/")
async def analyze_frame(file: UploadFile = File(...)):
    print("Received file:", file.filename, file.content_type)

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
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

    # Object detection for safety equipment
    try:
        obj_response = rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=75
        )
        for label in obj_response.get("Labels", []):
            name = label.get("Name", "")
            confidence = label.get("Confidence", 0)
            if any(keyword in name.lower() for keyword in safety_keywords):
                objects.append({"name": name, "confidence": confidence})
    except Exception as e:
        return {"error": f"Object detection error: {str(e)}"}

    result = "yes" if objects else "no"
    return {
        "detected_objects": objects,
        "result": result
    }
