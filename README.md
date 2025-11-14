
# Hand-Controlled 3D Model Viewer (Ursina + Mediapipe)

This project allows you to **control a 3D model using your hands** through your webcam using:

* **Ursina Engine** for rendering
* **Mediapipe Hands** for gesture detection
* **OpenCV** for webcam input

You can rotate, zoom, translate, pause, reset, and take screenshots — all with intuitive gestures.

---

## ✨ Features

### 🎮 Right Hand Controls

| Gesture                       | Action       |
| ----------------------------- | ------------ |
| Move index finger             | Rotate model |
| Pinch (thumb + index)         | Zoom in/out  |
| Peace ✌ (paused)              | Screenshot   |
| Perfect thumbs-up 👍 (paused) | Reset model  |

---

### ✋ Left Hand Controls

| Gesture           | Action               |
| ----------------- | -------------------- |
| Relaxed open palm | Pause                |
| Move wrist        | Translate/drag model |

---

### 🖼 Model Auto-Setup

* Automatically centered
* Automatically scaled
* Loads in perfect visible position

---

## 📁 Project Structure

Your model must be placed inside the **assets folder**:

```
your_project/
│
├── assets/
│   └── vintage_racing_car.glb
│
├── screenshots/
│   └── (auto-saved images)
│
└── main.py
```

In code, set the model path like:

```python
car = Entity(model='assets/vintage_racing_car.glb')
```

Ursina automatically loads models from the **assets folder**, so no special handling is needed.

---

## 💡 Gesture Details

### Pause

Left hand with **any 2+ fingers extended**.

### Zoom

Right-hand **pinch strength**:

* Strong pinch = zoom in
* Weak pinch = zoom out

### Rotation

Right-hand index finger movement rotates the 3D model.

### Translation

Left wrist movement drags the model in screen space.

### Reset

Perfect thumbs-up 👍 with the right hand while paused.

### Screenshot

Right-hand peace sign ✌ while paused.

---

## 🔧 Requirements

```
python 3.10 – 3.12
ursina
opencv-python
mediapipe
numpy
```

Install:

```bash
pip install ursina opencv-python mediapipe numpy
```

---

## ▶️ How to Run

```bash
python main.py
```

Webcam and 3D viewer windows will open.

---

## 🏗 Internals

* Mediapipe provides 21 hand landmarks
* Gestures derived from finger joint ordering and distances
* Zoom uses smooth pinch-strength processing
* Translation uses left wrist movement in **world space**
* Rotation uses right index finger movement
* All movements use smoothed filters to remove jitter
* Model bounds automatically analyzed to center and scale the GLB

---

## 📷 Screenshots

Saved inside:

```
screenshots/
```

Automatically named like:

```
snap_20240215_184200.png
```

---

## 🚨 Notes

* Good lighting improves hand tracking
* Keep hands 30–60 cm from webcam
* Reset & screenshot work **only when paused**
* Translation is world-space (not dependent on rotation)

---

