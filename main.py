import sys
import os

# Abaikan folder lokal './mediapipe' agar Python menggunakan pustaka site-packages resmi
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath('.')]

import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Daftar 11 Filter Retrolens
FILTERS = [
    "MONO", 
    "DUAL-TONE", 
    "PIXELATE", 
    "INVERT", 
    "SEPIA", 
    "BLUR", 
    "THERMAL", 
    "SKETCH", 
    "GLITCH", 
    "NEON", 
    "GALAXY"
]

# Palet Warna Neon / Cyberpunk (BGR)
COLOR_CYAN = (255, 240, 0)
COLOR_VIOLET = (255, 0, 189)
COLOR_GOLD = (0, 215, 255)
COLOR_EMERALD = (80, 220, 100)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (15, 12, 18)

def generate_galaxy_background(height=1080, width=1920):
    """Menghasilkan background Galaxy / Space Starfield secara prosedural"""
    galaxy_bg = np.zeros((height, width, 3), dtype=np.uint8)
    galaxy_bg[:] = (35, 12, 45) # Deep Space Violet

    # Bintang-bintang kecil
    for _ in range(900):
        sx = np.random.randint(0, width)
        sy = np.random.randint(0, height)
        galaxy_bg[sy, sx] = (255, 255, 255)

    # Bintang-bintang bercahaya lebih besar (Warna-warni neon)
    for _ in range(120):
        sx = np.random.randint(0, width)
        sy = np.random.randint(0, height)
        radius = np.random.randint(2, 6)
        color = (np.random.randint(180, 255), np.random.randint(100, 255), 255)
        cv2.circle(galaxy_bg, (sx, sy), radius, color, -1)

    return galaxy_bg

def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    """Menerapkan salah satu dari 11 filter pada Region of Interest (ROI)"""
    if roi is None or roi.size == 0:
        return roi

    h_r, w_r = roi.shape[:2]

    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)

    elif filter_name == "BLUR":
        ksize = max(3, (min(h_r, w_r) // 10) | 1) # Harus ganjil
        return cv2.GaussianBlur(roi, (ksize, ksize), 0)

    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255] # Neon Orange
        filtered[mask_c == 0] = [219, 0, 189]  # Neon Pink
        return filtered

    elif filter_name == "PIXELATE":
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r // 12), max(1, h_r // 12)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)

    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif filter_name == "GLITCH":
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi

    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = COLOR_CYAN
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)

    elif filter_name == "GALAXY" and frame_galaxy is not None:
        roi_galaxy = frame_galaxy[y:y+h_r, x:x+w_r]
        if roi_galaxy.shape[:2] == (h_r, w_r):
            if mask_person is not None:
                roi_mask = mask_person[y:y+h_r, x:x+w_r]
                bg_condition = (roi_mask == 0)
                filtered = roi.copy()
                filtered[bg_condition] = roi_galaxy[bg_condition]
                return filtered
            else:
                # Transparan blend galaxy background
                return cv2.addWeighted(roi, 0.45, roi_galaxy, 0.55, 0)

    return roi

def draw_sleek_hud(img, fps, current_filter_idx, gesture_text, full_frame_mode):
    """Menggambar HUD Glassmorphism Estetik & Modern di Bagian Atas Tampilan"""
    h, w, _ = img.shape
    overlay = img.copy()

    # --- 1. GLASSMORPHISM PANEL HEADER ATAS ---
    panel_h = 110
    panel_margin = 15
    cv2.rectangle(overlay, (panel_margin, panel_margin), (w - panel_margin, panel_h), COLOR_DARK_BG, -1)
    
    # Transparency blend untuk efek kaca (Glassmorphism)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Frame border neon halus
    cv2.rectangle(img, (panel_margin, panel_margin), (w - panel_margin, panel_h), COLOR_CYAN, 1, cv2.LINE_AA)

    # --- 2. STATUS FPS & BADGES ---
    # FPS Pill Indicator
    cv2.rectangle(img, (panel_margin + 15, panel_margin + 12), (panel_margin + 175, panel_margin + 42), (40, 40, 40), -1)
    cv2.circle(img, (panel_margin + 30, panel_margin + 27), 6, COLOR_EMERALD, -1)
    cv2.putText(img, f"30 FPS LOCKED", (panel_margin + 45, panel_margin + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    # Mode Indicator (Portal vs Full)
    mode_str = "MODE: FULL FRAME" if full_frame_mode else "MODE: 4-POINT PORTAL"
    mode_color = COLOR_GOLD if full_frame_mode else COLOR_CYAN
    cv2.rectangle(img, (panel_margin + 190, panel_margin + 12), (panel_margin + 410, panel_margin + 42), (40, 40, 40), -1)
    cv2.putText(img, mode_str, (panel_margin + 205, panel_margin + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

    # Status Gestur Text
    cv2.putText(img, f"Gestur: {gesture_text}", (panel_margin + 15, panel_margin + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.putText(img, "Ganti Filter: Touch Jempol+Kelingking | [TAB] Pindah | [f] Fullscreen | [q] Keluar", 
                (panel_margin + 15, panel_margin + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # --- 3. FILTER CAROUSEL PILLS (SELEKSI FILTER ESTETIK) ---
    start_x = panel_margin + 430
    pill_w = 115
    pill_h = 32
    max_visible = min(len(FILTERS), 6)

    for i in range(max_visible):
        # Tampilkan filter yang aktif dan sekitarnya
        f_idx = (current_filter_idx + i - 1) % len(FILTERS)
        px1 = start_x + i * (pill_w + 8)
        py1 = panel_margin + 12
        px2 = px1 + pill_w
        py2 = py1 + pill_h

        if px2 > w - panel_margin - 10:
            break

        is_active = (f_idx == current_filter_idx)
        bg_col = COLOR_VIOLET if is_active else (35, 35, 35)
        text_col = COLOR_WHITE if is_active else (160, 160, 160)

        cv2.rectangle(img, (px1, py1), (px2, py2), bg_col, -1)
        if is_active:
            cv2.rectangle(img, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), COLOR_GOLD, 2, cv2.LINE_AA)

        f_name = FILTERS[f_idx]
        if len(f_name) > 10:
            f_name = f_name[:9] + "."
        
        cv2.putText(img, f_name, (px1 + 8, py1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_col, 1, cv2.LINE_AA)

def main():
    # Ambil index kamera dari argumen terminal (default: 0, atau 1 untuk USB)
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0

    print(f"Mencoba membuka kamera index: {cam_index}...")
    cap = cv2.VideoCapture(cam_index)

    # Fallback ke kamera default (0) jika kamera USB tidak dapat dibuka
    if not cap.isOpened() and cam_index != 0:
        print(f"Kamera index {cam_index} tidak dapat dibuka, mengalihkan ke kamera default (0)...")
        cam_index = 0
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera/webcam apapun.")
        return

    # Target 30 FPS dan Resolusi Full HD 1920x1080
    TARGET_FPS = 30
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Inisialisasi Window Resizable agar pas di layar laptop
    window_name = "RETROLENS - Hand Gesture Portal Filter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Generate Galaxy Background Prosedural
    galaxy_bg = generate_galaxy_background(1080, 1920)

    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    current_filter = 0
    gesture_triggered = False
    full_frame_mode = False
    prev_time = time.time()

    print("\n========================================================")
    print(" 🚀 RETROLENS Estetik - Hand Gesture Portal Filters Ready!")
    print(" 📷 Resolusi: Full HD 1920x1080 @ 30 FPS Locked (120° Wide Angle)")
    print("========================================================\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while cap.isOpened():
            frame_start_time = time.time()

            success, img = cap.read()
            if not success:
                continue

            # Flip horizontal agar seperti cermin
            img = cv2.flip(img, 1)
            h, w, c = img.shape

            # Crop background galaxy sesuai dimensi aktif
            frame_galaxy = galaxy_bg[:h, :w]

            # Konversi BGR ke RGB untuk MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            filter_name = FILTERS[current_filter]
            pts_portal = []
            change_filter = False

            if results.multi_hand_landmarks:
                # 1. CEK GESTUR GANTI FILTER: 2 Ujung Telunjuk Saling Mendekat
                if len(results.multi_hand_landmarks) >= 2:
                    idx0 = results.multi_hand_landmarks[0].landmark[8]
                    idx1 = results.multi_hand_landmarks[1].landmark[8]
                    pt0 = (int(idx0.x * w), int(idx0.y * h))
                    pt1 = (int(idx1.x * w), int(idx1.y * h))
                    if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 45:
                        change_filter = True

                # 2. CEK GESTUR GANTI FILTER: Jempol & Kelingking Saling Mendekat
                for hand_lms in results.multi_hand_landmarks:
                    thumb = hand_lms.landmark[4]
                    pinky = hand_lms.landmark[20]
                    tx, ty = int(thumb.x * w), int(thumb.y * h)
                    px, py = int(pinky.x * w), int(pinky.y * h)
                    if math.hypot(tx - px, ty - py) < 45:
                        change_filter = True

                # Debounce Trigger Ganti Filter
                if change_filter:
                    if not gesture_triggered:
                        current_filter = (current_filter + 1) % len(FILTERS)
                        gesture_triggered = True
                        print(f"Gestur Terdeteksi! Pindah ke Filter: {FILTERS[current_filter]}")
                else:
                    gesture_triggered = False

                # Ambil koordinat Ujung Jempol (ID 4) dan Telunjuk (ID 8)
                for hand_lms in results.multi_hand_landmarks:
                    # Gambar koneksi rangka tangan
                    mp_drawing.draw_landmarks(
                        img,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    for id_lm in [4, 8]:
                        cx = int(hand_lms.landmark[id_lm].x * w)
                        cy = int(hand_lms.landmark[id_lm].y * h)
                        pts_portal.append([cx, cy])
                        
                        # Target Ring Estetik di Ujung Jari
                        cv2.circle(img, (cx, cy), 12, COLOR_CYAN, 2, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), 5, COLOR_GOLD, -1, cv2.LINE_AA)

                # ========================================================
                # MODE 1: FULL FRAME FILTER
                # ========================================================
                if full_frame_mode:
                    img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

                # ========================================================
                # MODE 2: 4-POINT PORTAL POLYGON (2 TANGAN)
                # ========================================================
                elif len(pts_portal) == 4:
                    pts_portal.sort(key=lambda p: p[1])
                    top_pts = pts_portal[:2]
                    bottom_pts = pts_portal[2:]

                    top_pts.sort(key=lambda p: p[0])
                    bottom_pts.sort(key=lambda p: p[0])

                    poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)

                    x, y, bw, bh = cv2.boundingRect(poly_pts)
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)

                    if bw > 0 and bh > 0:
                        roi = img[y:y+bh, x:x+bw].copy()
                        filtered_roi = apply_filter(roi, filter_name, x, y, None, frame_galaxy)

                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        poly_roi = poly_pts - [x, y]
                        cv2.fillPoly(mask, [poly_roi], 255)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        # Terapkan filter di dalam polygon
                        img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)

                        # Garis Tepi Neon Ganda Estetik
                        cv2.polylines(img, [poly_pts], True, COLOR_VIOLET, 4, cv2.LINE_AA)
                        cv2.polylines(img, [poly_pts], True, COLOR_CYAN, 2, cv2.LINE_AA)

                        # --- PARTIKEL BINTANG GLOW DI TEPI PORTAL ---
                        for i in range(4):
                            p_start = poly_pts[i]
                            p_end = poly_pts[(i + 1) % 4]
                            for _ in range(6):
                                alpha = np.random.random()
                                px = int(p_start[0] * alpha + p_end[0] * (1 - alpha)) + np.random.randint(-14, 14)
                                py = int(p_start[1] * alpha + p_end[1] * (1 - alpha)) + np.random.randint(-14, 14)
                                cv2.circle(img, (px, py), np.random.randint(2, 5), COLOR_GOLD, -1)

                        # Label Portal Neon
                        cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], max(35, top_pts[0][1] - 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_WHITE, 2, cv2.LINE_AA)

                # MODE 3: MINI PORTAL (1 TANGAN)
                elif len(pts_portal) == 2 and not full_frame_mode:
                    cx = (pts_portal[0][0] + pts_portal[1][0]) // 2
                    cy = (pts_portal[0][1] + pts_portal[1][1]) // 2
                    radius = int(math.hypot(pts_portal[0][0] - pts_portal[1][0], pts_portal[0][1] - pts_portal[1][1])) // 2 + 30

                    x1, y1 = max(0, cx - radius), max(0, cy - radius)
                    x2, y2 = min(w, cx + radius), min(h, cy + radius)
                    bw, bh = x2 - x1, y2 - y1

                    if bw > 0 and bh > 0:
                        roi = img[y1:y2, x1:x2].copy()
                        filtered_roi = apply_filter(roi, filter_name, x1, y1, None, frame_galaxy)

                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        cv2.circle(mask, (cx - x1, cy - y1), radius, 255, -1)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        img[y1:y2, x1:x2] = np.where(mask_3ch == 255, filtered_roi, roi)
                        cv2.circle(img, (cx, cy), radius, COLOR_CYAN, 3, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), radius + 4, COLOR_VIOLET, 1, cv2.LINE_AA)
                        cv2.putText(img, f"MINI PORTAL: {filter_name}", (x1, max(35, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)

            elif full_frame_mode:
                img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

            # ========================================================
            # 30 FPS FRAME PACING (LOCKED & SMOOTH)
            # ========================================================
            curr_time = time.time()
            elapsed = curr_time - frame_start_time
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)

            actual_fps = 1.0 / (time.time() - frame_start_time)

            gesture_status = f"[{FILTERS[current_filter]}] Status: Aktif" if gesture_triggered else "Siap / Bebas"

            # Tampilkan HUD Glassmorphism Estetik
            draw_sleek_hud(img, actual_fps, current_filter, gesture_status, full_frame_mode)

            # Tampilkan Window Kamera
            cv2.imshow(window_name, img)

            # Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key in [9, 32]: # TAB / SPACE
                current_filter = (current_filter + 1) % len(FILTERS)
            elif key == ord('f') or key == ord('F'):
                full_frame_mode = not full_frame_mode
            elif ord('0') <= key <= ord('9'):
                idx_key = key - ord('0')
                if idx_key < len(FILTERS):
                    current_filter = idx_key

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
