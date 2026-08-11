# 🖐️ Hand Gesture Portal Filters (RETROLENS)

A real-time Computer Vision application built with **OpenCV** and **Google MediaPipe** that creates dynamic, foldable 3D camera filter portals using real-time hand tracking.

---

## 🌟 Key Features

- **11 Retro Filters**: `MONO`, `DUAL-TONE`, `PIXELATE`, `INVERT`, `SEPIA`, `BLUR`, `THERMAL`, `SKETCH`, `GLITCH`, `NEON`, `GALAXY`
- **Foldable 4-Point Portal**: Form mid-air portals using your hands (Thumb & Index) with deterministic hand mapping and smooth EMA vertex interpolation.
- **Dual-Portal Layering**: Form 2 independent filter portals simultaneously (Portal 1: Thumb+Index, Portal 2: Index+Middle).
- **3D Volumetric & Chromatic Effect**: Overlapping portals form a 3D wireframe prism with 3D RGB-shift (Chromatic Aberration) holographic blending.
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
