import React, { useEffect, useRef } from "react";
import GestureRecognition from "./components/GestureRecognition";
import "./app.css"; // Import CSS file

const App = () => {
    const videoRef = useRef(null);

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            })
            .catch((error) => console.error("Error accessing webcam:", error));
    }, []);

    return (
        <div className="app-container">
            <h1 className="title">Hand Gesture Recognition</h1>
            <GestureRecognition />
            <video ref={videoRef} autoPlay className="webcam-video" />
        </div>
    );
};

export default App;
