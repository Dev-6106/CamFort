# CamFort 
**Real-Time Sign Language Detection System**

---

## Overview

Camfort is a team project developed as part of the Object-Oriented Programming (OOP) coursework.  
It aims to bridge the communication gap between the deaf and hearing communities by recognizing **American Sign Language (ASL)** gestures using a webcam.  

By leveraging **MediaPipe** and **OpenCV**, Camfort detects hand landmarks in real-time and identifies the corresponding alphabet letters (A–Z) based on geometric relationships between finger positions.

---

## Features

- Detects ASL gestures in real-time using webcam input  
- Recognizes letters from A to Z (static gestures)  
- Converts detected gestures into readable text  
- Allows word building through detected letters  
- Displays the recognized letter on live video feed  
- Easily extendable for new gestures or sign languages  

---

## Tech Stack

**Programming Language:** Python  
**Libraries Used:**
- OpenCV (for image and video processing)  
- MediaPipe (for hand tracking and landmark detection)  
- NumPy (for mathematical computations)  

**Concepts Used:**
- Object-Oriented Programming (OOP) principles  
- Real-time computer vision  
- Geometric analysis for gesture recognition  

---

## Team Members

- Jvel Kothiya (U24CS025)
- Jay Patel (U24CS026)
- Tarun Vadivelan (U24CS027)
- Mithesh Manas (U24CS030)
- Dhruv Sarvaiya (U24CS034)
- Rupesh Chaudhary (U24CS035)
- Nihar Mehta (U24CS036)
- Dev Shyara (U24CS056)
- Rudra Patni (U24CS074)
- Aryan Mori (U24CS080)

---

## Working Principle

The project works on the concept of detecting **21 key landmarks** on a human hand using the **MediaPipe Hands** module.  
Each letter of ASL is identified based on the **relative positions, distances, and angles** between these landmarks.

### Core Logic:
1. Capture frames from webcam using OpenCV.  
2. Use MediaPipe to detect hand landmarks.  
3. Apply geometric rules (angles, distances) to match the hand posture with a specific ASL letter.  
4. Display the detected letter on the live video feed.  
5. Combine detected letters into words interactively.

---

## Object-Oriented Design

The system follows **OOP principles** for modularity and maintainability.

### Class Used:
**`ASLDetector`**
- **Attributes:**
  - Hand detection setup using MediaPipe  
  - Landmark history for gesture tracking  

- **Methods:**
  - `calculate_angle(a, b, c)` – Computes the angle between three points  
  - `detect_asl_letter()` – Determines which ASL letter the hand represents  
  - `process_frame()` – Processes each frame and overlays detected letters  

---

## Instructions to Run

1. Run the Python file (`camfort.py`) in any IDE such as VS Code or PyCharm.  
2. Ensure your **webcam is connected and accessible**.  
3. Show any ASL gesture (A–Z) to the camera.  
4. Press the following keys while the program is running:
   - **Space**: Add the current letter to the word  
   - **C**: Clear the current word  
   - **Q**: Quit the program  

---

## Screenshots

### Input:
<img width="616" height="497" alt="image" src="https://github.com/user-attachments/assets/c24e22ce-0ef9-4ba2-b80d-860bb46c0dae" />

### Output:
<img width="703" height="157" alt="image" src="https://github.com/user-attachments/assets/5c1eb486-75fe-4616-8a6a-a48043f1f66c" />

---

## Future Scope

- Implement dynamic gesture detection (for letters “J” and “Z”)  
- Extend recognition to **Indian Sign Language (ISL)**  
- Add **speech synthesis** for detected words  
- Develop a **web or mobile version** for accessibility  

---

## Conclusion

Camfort demonstrates the integration of **Object-Oriented Programming concepts** with **computer vision** and **AI-based gesture recognition**.  
It showcases how modern programming techniques can be applied to create impactful, real-world assistive technologies.

---

## Acknowledgment

We would like to express our sincere gratitude to our faculty and mentors for their continuous guidance and support throughout this project.

---
