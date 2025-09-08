# webcam_test.py
# Real-time traffic sign recognition with image processing

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from labels import sign_names

MODEL_PATH = "model/traffic_sign_cnn.h5"
IMG_SIZE = (32, 32)

def preprocess_frame(frame_bgr):
    """
    Apply image processing techniques:
    - Histogram equalization
    - Gaussian blur
    - Grayscale conversion
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    frame_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame_bgr = cv2.GaussianBlur(frame_bgr, (3,3), 0)
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Resize and normalize
    img = cv2.resize(frame_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("[info] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess_frame(frame)
        probs = model.predict(inp, verbose=0)[0]
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        label = sign_names.get(cls, f"Class {cls}")

        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 3)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60,255,60), 2)

        cv2.imshow("GTSRB - Webcam Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
