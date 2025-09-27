import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ============================
# Load trained model & classes
# ============================
model = tf.keras.models.load_model("models/gesture_recognition_model.h5")
class_names = np.load("gesture_classes.npy", allow_pickle=True)

print("✅ Model and class names loaded successfully!")
print(f"🖐️ Classes: {class_names}")

# ============================
# Initialize MediaPipe Hands
# ============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ============================
# Open webcam
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access webcam")
    exit()

print("🎥 Starting Real-Time Gesture Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, exiting...")
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(max(min([lm.x for lm in hand_landmarks.landmark]) * w, 0))
            y_min = int(max(min([lm.y for lm in hand_landmarks.landmark]) * h, 0))
            x_max = int(min(max([lm.x for lm in hand_landmarks.landmark]) * w, w))
            y_max = int(min(max([lm.y for lm in hand_landmarks.landmark]) * h, h))

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess hand image
            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Prediction
            prediction = model.predict(hand_img, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            gesture = class_names[predicted_class]

            # Display gesture + confidence
            cv2.putText(frame, f"{gesture} ({confidence:.2f})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("🤖 Real-Time Gesture Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
