# 🖐️ Hand Gesture Portal Filters (RETROLENS)

A real-time Computer Vision application built with **OpenCV** and **Google MediaPipe** that creates dynamic, foldable 3D camera filter portals using real-time hand tracking.

---

## 🌟 Key Features

- **11 Retro Filters**: `MONO`, `DUAL-TONE`, `PIXELATE`, `INVERT`, `SEPIA`, `BLUR`, `THERMAL`, `SKETCH`, `GLITCH`, `NEON`, `GALAXY`
- **Foldable 4-Point Portal**: Form mid-air portals using your hands with deterministic hand mapping and smooth EMA vertex interpolation.
- **Triple-Portal 3D System**: Form 3 independent filter portals simultaneously (Portal 1: Thumb+Index, Portal 2: Index+Middle, Portal 3: Thumb+Middle).
- **Full 3D Volumetric Mesh**: All 3 portals are connected in real-time by a 3D wireframe depth mesh, creating a complete floating 3D volumetric prism in mid-air with 3D RGB-shift (Chromatic Aberration) holographic blending.
- **Gesture Control**: Touch thumb to pinky or touch index fingertips to switch filters instantly.
- **Clean Instagram-Style UI**: Minimalist bottom filter carousel, sharp HD text, locked 30 FPS playback.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Default camera (index 0)
python main.py

# External USB camera (index 1)
python main.py 1
```

---

## ⌨️ Controls

| Key / Gesture | Action |
| --- | --- |
| `Pinch` (Thumb + Pinky / 2 Index) | Switch filter |
| `TAB` / `SPACE` | Next filter |
| `F` | Toggle Portal / Full-Frame mode |
| `0` - `9` | Select filter directly |
| `Q` / `ESC` | Quit |
