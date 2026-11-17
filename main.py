# uvicorn main:app --reload --host 0.0.0.0 --port 8000
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
import base64
import io
from PIL import Image

app = FastAPI(title="Face Mask Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and detector once at startup
detector = MTCNN()
model = load_model("FaceMaskClassifier.keras")
class_labels = ["with_mask", "without_mask"]


LABEL_META = {
    "without_mask": {
        "display": "No Mask",
        "color": (0, 0, 255),       # BGR red
        "hex": "#ef4444",
    },
    "with_mask": {
        "display": "Mask On",
        "color": (0, 200, 80),      # BGR green
        "hex": "#22c55e",
    }
}


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to a BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def encode_image(frame: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    _, buffer = cv2.imencode(".png", frame)
    return base64.b64encode(buffer).decode("utf-8")


def analyze_frame(frame: np.ndarray):
    """Run MTCNN face detection + mask classification on a single frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    results = []

    for face in faces:
        x, y, w, h = face["box"]
        # Clamp to image bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        face_crop = frame[y:y2, x:x2]
        if face_crop.size == 0:
            continue

        face_resized = cv2.resize(face_crop, (224, 224))
        face_input = np.expand_dims(face_resized, axis=0)
        face_input = preprocess_input(face_input.astype(np.float32))

        probs = model.predict(face_input, verbose=0)[0]
        predicted_idx = int(np.argmax(probs))
        label_key = class_labels[predicted_idx]
        meta = LABEL_META[label_key]
        confidence = float(face["confidence"])
        mask_confidence = float(probs[predicted_idx])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), meta["color"], 2)

        # Label background pill
        label_text = f"{meta['display']}  {mask_confidence * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx, ly = x, y - 10
        cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 4), meta["color"], -1)
        cv2.putText(
            frame,
            label_text,
            (lx + 4, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "box": {"x": x, "y": y, "w": x2 - x, "h": y2 - y},
                "label": label_key,
                "display_label": meta["display"],
                "color": meta["hex"],
                "face_confidence": round(confidence, 4),
                "mask_confidence": round(mask_confidence, 4),
                "probabilities": {
                    class_labels[i]: round(float(probs[i]), 4) for i in range(len(class_labels))
                },
            }
        )

    return frame, results


@app.get("/")
def root():
    return {"status": "Face Mask Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accept an image file, run face detection + mask classification,
    return annotated image (base64) and per-face JSON results.
    """
    try:
        contents = await file.read()
        frame = decode_image(contents)

        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        annotated_frame, face_results = analyze_frame(frame)
        encoded_image = encode_image(annotated_frame)

        summary = {
            "total_faces": len(face_results),
            "no_mask": sum(1 for f in face_results if f["label"] == "without_mask"),
            "with_mask": sum(1 for f in face_results if f["label"] == "with_mask")
        }

        return {
            "annotated_image": encoded_image,
            "faces": face_results,
            "summary": summary,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
