import React, { useRef, useState } from "react";
import axios from "axios";

const GestureRecognition = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [gesture, setGesture] = useState("");

    const captureImage = () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL("image/jpeg").split(",")[1];

        // Send to Flask API
        axios.post("http://127.0.0.1:5000/predict", { image: imageData })
            .then((response) => {
                setGesture(response.data.gesture);
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    };

    return (
        <div className="container">
            <h2>Hand Gesture Recognition</h2>
            <video ref={videoRef} autoPlay width="300" height="200" />
            <canvas ref={canvasRef} width="64" height="64" style={{ display: "none" }} />
            <button onClick={captureImage}>Predict Gesture</button>
            <h3>Detected Gesture: {gesture}</h3>
        </div>
    );
};

export default GestureRecognition;
