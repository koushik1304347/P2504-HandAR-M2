# P2504-HandAR-M2

A real-time interactive 3D cube visualization controlled entirely using **hand gestures** — powered by **MediaPipe**, **OpenCV**, **PyOpenGL**, and **Pygame**.

You can rotate, zoom, and translate the cube naturally using both hands, with each hand performing different actions.

---

## ✨ Features

- 🧠 **Real-time Hand Tracking** using [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- 🖐️ **Dual-Hand Control**:
  - **Right Hand** → Rotate & Zoom
  - **Left Hand** → Translate (Move the cube)
- 🧊 **3D Rendering** using PyOpenGL and Pygame
- 🎨 **Colored Faces with Transparency** for a modern look
- ⚙️ **Smooth Interactive Controls** with continuous tracking at 60 FPS

---

## 🕹️ Controls

| Hand | Action | Gesture |
|------|---------|----------|
| 🖐️ Right Hand | Rotate | Move index finger (left/right/up/down) |
| 🖐️ Right Hand | Zoom | Pinch (thumb + index closer/farther) |
| ✋ Left Hand | Translate | Move index finger in any direction |

> 💡 Tip: Keep both hands visible in the webcam frame for the best experience.

---

## 🧰 Requirements

Make sure you have **Python 3.8+** installed, then install these dependencies:

```bash
pip install opencv-python mediapipe pygame PyOpenGL PyOpenGL_accelerate
