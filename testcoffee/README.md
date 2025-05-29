# 8VC Hackathon

**Real-time Visual Servoing & Language-Guided Object Manipulation on Hello Robot Stretch**

A proof-of-concept system that turns a Hello Robot Stretch into an intelligent, vision-guided manipulator. We fuse state-of-the-art computer vision (YOLOv8, Segment Anything Model, ArUco), a lightweight LLaMA‐based reasoning engine, and Intel RealSense D405 depth sensing to enable:

- **Object Detection & Segmentation**  
  - YOLOv8 for fast bounding-box detection  
  - Meta’s SAM for precise pixel-level masks  
- **Marker-based Calibration**  
  - ArUco markers to localize camera & robot end-effector in a common frame  
- **Natural-Language Task Planning**  
  - LLaMA-based “brain” chooses sorting orders and high-level instructions  
- **Real-time Visual Servoing**  
  - Velocity‐based control loop drives Stretch to align & grasp  
  - Smooth motion thanks to normalized velocity control  
- **Modular Networking**  
  - Zero-MQ sockets stream RGB & depth frames between Stretch and inference server  
  - Asynchronous loops decouple sensing, perception, planning, and acting  

---

## 🚀 Quick Start

1. **Clone & enter**  
   ```bash
   git clone https://github.com/yalcintur/8VC-Hackathon.git
   cd 8VC-Hackathon
