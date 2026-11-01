                                                    AI-Based Real-Time Driver Monitoring System
                                                    
📌 Project Overview
This project is a real-time computer vision application designed to enhance road safety by continuously monitoring driver attentiveness. Built with Python and OpenCV, the system leverages Google's MediaPipe Face Mesh to track 468 3D facial landmarks in real time. By analyzing continuous facial geometry and head poses using NumPy, the system accurately detects physiological signs of fatigue and distraction, immediately alerting the driver via a multi-threaded, zero-latency audio alarm.

🚀 Key Features
Drowsiness & Microsleep Detection: Calculates the continuous Eye Aspect Ratio (EAR) across 12 distinct eye landmarks to detect prolonged eye closures and heavy eyelids.

Yawn Frequency Tracking: Computes the Mouth Aspect Ratio (MAR) to identify deep yawning, a primary indicator of driver fatigue.

Head Pose Distraction Monitoring: Analyzes dynamic spatial ratios between the nose tip and lateral cheek landmarks to detect hard head turns and prolonged visual distraction from the road.

Zero-Latency Audio Alarms: Implements an asynchronous, multi-threaded background alarm system that triggers cross-platform audio warnings without degrading the main video processing framerate.

CPU-Optimized Inference: Engineered for high-efficiency, real-time processing without requiring dedicated GPU hardware, serving as a foundational architecture for future embedded system deployment.

🛠️ Technology Stack
Language: Python

Computer Vision: OpenCV (cv2)

Machine Learning / AI: Google MediaPipe (Face Mesh)

Mathematics & Matrix Operations: NumPy

Architecture: Multi-threading (Python threading module)
