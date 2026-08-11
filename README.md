# 🖐️ Hand Gesture Portal Filters (RETROLENS)

A real-time Computer Vision application built with **OpenCV** and **Google MediaPipe** that creates dynamic camera filter portals using hand tracking gestures.

---

## 🌟 Features

- **11 Retro Filters**: `MONO`, `DUAL-TONE`, `PIXELATE`, `INVERT`, `SEPIA`, `BLUR`, `THERMAL`, `SKETCH`, `GLITCH`, `NEON`, `GALAXY`
- **4-Point Hand Portal**: Form a 4-point polygon using two hands (thumb and index fingertips) to cast filter portals in mid-air.
- **Gesture Controls**: Touch thumb to pinky or touch two index fingertips (< 45px) to switch filters seamlessly.
- **Minimalist HD Overlay**: Clean Instagram-style carousel pills at the bottom, locked 30 FPS display, unobtrusive corner HUD.
- **1080p @ 30 FPS Optimized**: Configured for Full HD wide-angle webcams with smooth MJPG streaming.

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

## ⌨️ Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `TAB` / `SPACE` | Switch to next filter |
| `F` | Toggle between Portal mode & Full Frame mode |
| `0` - `9` | Directly select a filter |
| `Q` / `ESC` | Exit application |

---

## 📄 License
MIT License
