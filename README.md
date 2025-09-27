# Hand Gestures Recognition with AI

## Overview

**Hand Gestures Recognition with AI** is a real-time system that detects and interprets hand gestures using computer vision and deep learning. This project enables intuitive human-computer interaction, allowing users to control applications, devices, or virtual interfaces without physical input devices.

By combining **OpenCV** for image processing and **TensorFlow/Keras** for deep learning, the system achieves accurate and responsive gesture recognition.

---

## Features

* **Real-time Detection:** Track and recognize hand gestures live using your webcam.
* **AI-Powered Recognition:** TensorFlow-based neural networks ensure high accuracy in gesture classification.
* **Customizable Gestures:** Easily extendable to add new gestures or modify existing ones.
* **Cross-Platform:** Compatible with Windows, Linux, and macOS.
* **Interactive Applications:** Integrates seamlessly with games, presentations, and robotics projects.

---

## Screenshots

**Real-time Hand Gesture Detection Interface:**
![1758177907917](https://github.com/user-attachments/assets/9d4064de-8e62-4fd9-8e22-6a213d70ab7a)

**Gesture Classification Output:**
![1758177907240](https://github.com/user-attachments/assets/da10bea5-f73c-4a17-9389-0153788decf1)

---

## Technologies Used

* **Python** – Core programming language
* **OpenCV** – Image and video processing
* **TensorFlow / Keras** – Deep learning model development
* **NumPy** – Numerical operations

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mohanveeramanikantak/Hand-Gestures-Recognition-with-AI.git
```

### 2. Navigate into the project directory

```bash
cd Hand-Gestures-Recognition-with-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

1. Preprocess the dataset:

```bash
python preprocess.py
```

2. Train the model:

```bash
python train_model.py
```

3. Start gesture recognition:

```bash
python predict.py
```

---

## Usage

1. Launch the main Python script (`predict.py`).
2. Position your hand in front of the webcam.
3. The system will detect and classify gestures in real-time.
4. Add or modify gesture classes as needed to suit your application.

---

## Contributing

Contributions are welcome! You can help by:

* Adding new gestures
* Improving model accuracy
* Enhancing UI/UX
* Reporting bugs or suggesting features

Please feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

