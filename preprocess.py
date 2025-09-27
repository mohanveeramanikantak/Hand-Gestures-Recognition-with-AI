import os
import cv2
import numpy as np

# Path to your dataset folder (update this)
dataset_path = r"C:\Users\USER\Desktop\Hand Gesture Recognition Using AI\Dataset\leapGestRecog\00"

# Image size for resizing
IMG_SIZE = 128  # 128x128 pixels

# Lists to store image data and labels
data = []
labels = []
class_names = []  # To store gesture names

# Load images from each folder
for label, folder_name in enumerate(sorted(os.listdir(dataset_path))):
    folder_path = os.path.join(dataset_path, folder_name)

    if not os.path.isdir(folder_path):
        continue  # Skip non-folder files

    class_names.append(folder_name)  # Save class name

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Warning: Unable to read {img_path}")
            continue

        # Resize and normalize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize pixels (0-1)

        # Append to dataset
        data.append(img)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Save processed dataset
np.save("hand_gesture_data.npy", data)
np.save("hand_gesture_labels.npy", labels)
np.save("gesture_classes.npy", class_names)  # Save gesture names

print("✅ Dataset preprocessing completed!")
print(f"📸 Total images processed: {len(data)}")
print(f"🖐️ Classes found: {class_names}")
