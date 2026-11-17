
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from mtcnn import MTCNN
import base64
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input

class Image(BaseModel):
    image: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # all origins allowed
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model
model = load_model('face_mask_classifier1.keras') 
print('model loaded')

def base64_to_frame(base64_string):
    # Step 1: Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Step 2: Convert bytes to numpy array
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Step 3: Decode image (BGR format)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return frame

detector = MTCNN()


@app.post("/predict/")
def predict(imgData: Image):
    print(imgData.image) # base64 image
    frame = base64_to_frame(imgData.image)

    # detect faces using MTCNN
    faces = detector.detect_faces(frame)
    
    predicted = 'masked'
    return {"predicted": predicted}