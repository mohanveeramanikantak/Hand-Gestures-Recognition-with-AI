import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

# ============================
# Load Preprocessed Data
# ============================
X = np.load("hand_gesture_data.npy")  # Image data
y = np.load("hand_gesture_labels.npy")  # Labels
class_names = np.load("gesture_classes.npy", allow_pickle=True)  # Gesture names

print(f"✅ Dataset loaded successfully!")
print(f"📸 Images: {X.shape}, 🏷️ Labels: {y.shape}, 🖐️ Classes: {len(class_names)}")
print(f"Classes: {class_names}")

# ============================
# Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")

# ============================
# Build CNN Model
# ============================
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')  # Output = number of classes
])

# ============================
# Compile Model
# ============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================
# Train Model
# ============================
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=3, min_delta=0.001, restore_best_weights=True)]
)

# ============================
# Save Model
# ============================
os.makedirs("models", exist_ok=True)
model.save("models/gesture_recognition_model.h5")

print("✅ Model training complete!")
print("📂 Saved as 'models/gesture_recognition_model.h5'")

# ============================
# Evaluate on Test Set
# ============================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📈 Final Test Accuracy: {test_acc:.4f}")
