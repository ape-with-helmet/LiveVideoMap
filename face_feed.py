import cv2
import numpy as np
from mtcnn import MTCNN
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras


model = load_model(r'models/facenet_keras.h5')  # Ensure you have the pre-trained FaceNet model file in the 'models' folderimport tensorflow as tf

print(tf.__version__)
print(keras.__version__)
# Initialize MTCNN face detector# Load the FaceNet model using tensorflow.keras
detector = MTCNN()

def capture_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Live Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(frame):
    result = detector.detect_faces(frame)
    face_locations = [(r['box'][1], r['box'][0] + r['box'][2], r['box'][1] + r['box'][3], r['box'][0]) for r in result]
    return face_locations

def capture_aadhaar_face():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_locations = detect_faces(frame)
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow('Aadhaar Card', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save the image
            for top, right, bottom, left in face_locations:
                aadhaar_face = frame[top:bottom, left:right]
                cv2.imwrite('aadhaar_face.jpg', aadhaar_face)
                break
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'aadhaar_face.jpg'

def preprocess_face(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_face = preprocess_face(image)
    face_embedding = model.predict(preprocessed_face)
    return face_embedding[0]

def euclidean_distance(embedding1, embedding2):
    return norm(embedding1 - embedding2)

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = norm(embedding1)
    norm2 = norm(embedding2)
    return dot_product / (norm1 * norm2)

def match_faces(aadhaar_face_embedding, live_face_embedding):
    euclidean_dist = euclidean_distance(aadhaar_face_embedding, live_face_embedding)
    cosine_sim = cosine_similarity(aadhaar_face_embedding, live_face_embedding)
    return euclidean_dist, cosine_sim

def main():
    aadhaar_face_path = capture_aadhaar_face()
    aadhaar_face_embedding = get_face_embedding(aadhaar_face_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_locations = detect_faces(frame)
        for top, right, bottom, left in face_locations:
            live_face = frame[top:bottom, left:right]
            live_face_embedding = model.predict(preprocess_face(cv2.cvtColor(live_face, cv2.COLOR_BGR2RGB)))
            euclidean_dist, cosine_sim = match_faces(aadhaar_face_embedding, live_face_embedding[0])
            if euclidean_dist < 0.6:  # Threshold for Euclidean distance
                match_text = f"Match: Euclidean {euclidean_dist:.2f}, Cosine {cosine_sim:.2f}"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                match_text = f"No Match: Euclidean {euclidean_dist:.2f}, Cosine {cosine_sim:.2f}"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, match_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Live Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
