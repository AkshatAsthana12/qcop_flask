from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import boto3
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rekognition = boto3.client(
    'rekognition',
    region_name='us-east-1'  # or whatever region you're using
)

collection_id = "new_face_collection"

image_to_name = {
    "imag1.jpg": "Akshat",
    "photo.jpg": "Akshat"
}

safety_keywords = ['helmet', 'hardhat', 'safety vest', 'vest', 'goggles', 'boots', 'gloves']

@app.post("/analyze/")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    objects = []
    faces = []

    try:
        obj_response = rekognition.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=75
        )
        for label in obj_response.get("Labels", []):
            name = label["Name"]
            confidence = label["Confidence"]
            if any(keyword in name.lower() for keyword in safety_keywords):
                objects.append({"name": name, "confidence": confidence})

    except Exception as e:
        return {"error": f"Object detection error: {str(e)}"}

    try:
        face_response = rekognition.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': img_bytes},
            FaceMatchThreshold=95,
            MaxFaces=5
        )
        for match in face_response.get("FaceMatches", []):
            face = match["Face"]
            face_id = face["ExternalImageId"]
            similarity = match["Similarity"]
            name = image_to_name.get(face_id, "Person")
            faces.append({"name": name, "similarity": similarity})
    except Exception as e:
        return {"error": f"Face recognition error: {str(e)}"}

    return {
        "detected_objects": objects,
        "recognized_faces": faces
    }
