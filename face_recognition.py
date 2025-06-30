import boto3
import cv2
import time
from botocore.exceptions import ClientError

image_to_name = {
    "imag1.jpg": "Akshat",
    "photo.jpg": "Akshat"
}

def create_collection_if_not_exists(collection_id):
    client = boto3.client('rekognition')
    try:
        response = client.create_collection(CollectionId=collection_id)
        print(f'✅ Collection {collection_id} created: {response}')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            print(f'ℹ️ Collection {collection_id} already exists.')
        else:
            print(f'❌ Error creating collection: {e}')
            raise

def add_faces_to_collection(bucket, photo, collection_id):
    client = boto3.client('rekognition', region_name='us-east-1')
    print(f"\n📤 Adding face to collection '{collection_id}'")
    print(f"   ➤ Bucket: {bucket}")
    print(f"   ➤ Object Key: {photo}")
    try:
        response = client.index_faces(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
            ExternalImageId=photo,
            DetectionAttributes=['ALL']
        )
        print(f"✅ Face added from '{photo}' with response:")
        print(response)
    except ClientError as e:
        print(f"❌ Failed to add face from '{photo}': {e}")
        raise

def recognize_objects_and_faces(image, collection_id):
    client = boto3.client('rekognition')
    img_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    print("🔍 Detecting objects...")
    try:
        response_objects = client.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=75
        )
    except Exception as e:
        print(f"❌ Error in object detection: {e}")
        return [], []

    objects = []
    for label in response_objects.get('Labels', []):
        name = label['Name']
        confidence = label['Confidence']
        print(f'   ➤ Detected object: {name} with confidence: {confidence:.2f}')
        objects.append((name, confidence))

    print("🧠 Matching faces...")
    try:
        response_faces = client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': img_bytes},
            FaceMatchThreshold=95,
            MaxFaces=5
        )
    except Exception as e:
        print(f"❌ Error in face recognition: {e}")
        return objects, []

    faces = []
    for match in response_faces.get('FaceMatches', []):
        face = match['Face']
        face_id = face['ExternalImageId']
        name = image_to_name.get(face_id, "Person")
        print(f'   ➤ Matched face: {name} with similarity: {match["Similarity"]:.2f}')
        faces.append((name, match['Similarity']))

    return objects, faces

def main():
    bucket = "qcopbucket1"
    photos = ["imag1.jpg", "photo.jpg"]
    collection_id = "new_face_collection"

    print("🚀 Starting Rekognition system...\n")
    create_collection_if_not_exists(collection_id)

    for photo in photos:
        add_faces_to_collection(bucket, photo, collection_id)

    print("\n📸 Starting camera stream for real-time recognition...")

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    objects, recognized_faces = [], []

    # Locking and animation state
    recognized_face_lock = None
    recognized_obj_lock = None
    lock_time = 0
    face_match_streak = {}
    scan_animation_index = 0
    scan_lines = ["Scanning.", "Scanning..", "Scanning..."]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from webcam.")
            break

        frame_count += 1
        current_time = time.time()

        if current_time - lock_time > 3:
            recognized_face_lock = None
            recognized_obj_lock = None

        if frame_count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0:
                print("\n🧠 No faces detected locally. Sending full frame for recognition...")
                objects, detected_faces = recognize_objects_and_faces(frame, collection_id)
            else:
                print(f"\n👤 Detected {len(faces)} face(s). Processing face regions...")
                objects, detected_faces = [], []
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_region = frame[y:y+h, x:x+w]
                    obj, faces_in_region = recognize_objects_and_faces(face_region, collection_id)
                    objects.extend(obj)
                    detected_faces.extend(faces_in_region)

            for name, similarity in detected_faces:
                if similarity >= 95:
                    face_match_streak[name] = face_match_streak.get(name, 0) + 1
                    if face_match_streak[name] >= 2 and recognized_face_lock is None:
                        recognized_face_lock = (name, similarity)
                        lock_time = time.time()
                        face_match_streak = {}
                else:
                    face_match_streak[name] = 0

            safety_keywords = ['helmet', 'hardhat', 'safety vest', 'vest', 'goggles', 'boots', 'gloves']
            for obj_name, confidence in objects:
                if any(keyword in obj_name.lower() for keyword in safety_keywords) and confidence >= 75:
                    if recognized_obj_lock is None:
                        recognized_obj_lock = (obj_name, confidence)
                        lock_time = time.time()

        if recognized_obj_lock:
            cv2.putText(frame, f'{recognized_obj_lock[0]} ({recognized_obj_lock[1]:.2f}%)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, scan_lines[scan_animation_index % len(scan_lines)],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        if recognized_face_lock:
            cv2.putText(frame, f'{recognized_face_lock[0]} ({recognized_face_lock[1]:.2f}%)',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, scan_lines[scan_animation_index % len(scan_lines)],
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        scan_animation_index += 1

        cv2.imshow('🎥 Live Detection - Safety Gear & Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting stream...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released and all windows closed.")

if __name__ == "__main__":
    main()
