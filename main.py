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

def generate_galaxy_background(height=1080, width=1920):
    """Menghasilkan background Galaxy / Space Starfield secara prosedural"""
    galaxy_bg = np.zeros((height, width, 3), dtype=np.uint8)
    galaxy_bg[:] = (30, 10, 40) # Space purple base

    # Bintang-bintang kecil (Bintik putih)
    for _ in range(800):
        sx = np.random.randint(0, width)
        sy = np.random.randint(0, height)
        galaxy_bg[sy, sx] = (255, 255, 255)

    # Bintang-bintang bercahaya lebih besar (Warna-warni neon)
    for _ in range(100):
        sx = np.random.randint(0, width)
        sy = np.random.randint(0, height)
        radius = np.random.randint(2, 6)
        color = (np.random.randint(150, 255), np.random.randint(100, 255), 255)
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
        ksize = max(3, (min(h_r, w_r) // 10) | 1) # ganjil
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
        filtered[mask_c == 255] = [0, 165, 255] # Orange
        filtered[mask_c == 0] = [147, 20, 255]  # Pink
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
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
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
                return cv2.addWeighted(roi, 0.4, roi_galaxy, 0.6, 0)

    return roi

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

    # Set format MJPEG, resolusi Full HD 1920x1080 @ 30 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Inisialisasi Window Resizable agar pas di layar laptop
    window_name = "RETROLENS - Hand Gesture Portal Filter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Generate Galaxy Background
    galaxy_bg = generate_galaxy_background(1080, 1920)

    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    current_filter = 0
    gesture_triggered = False
    full_frame_mode = False # Toggle layar penuh vs Portal miring
    prev_time = 0

    print("\n========================================================")
    print(" 🚀 RETROLENS - Hand Gesture Portal Filters Ready!")
    print(" 📷 Mode Kamera: Full HD 1920x1080 @ 30 FPS (120° Wide Angle)")
    print(" ------------------------------------------------------")
    print(" 🖐️ GESTUR KONTROL PORTAL:")
    print("   - Bentuk Portal: Posisikan 2 Tangan (Jempol & Telunjuk)")
    print("   - Ganti Filter: Sentuhkan Jempol & Kelingking ATAU 2 Telunjuk")
    print(" ------------------------------------------------------")
    print(" ⌨️  KONTROL KEYBOARD:")
    print("   [TAB / SPACE] Ganti Filter berikutnya")
    print("   [f] Toggle Layar Penuh vs Portal Area")
    print("   [0 - 9] Pilih Filter Langsung (MONO, DUAL-TONE, SKETCH, dll)")
    print("   [q / ESC] Keluar dari Aplikasi")
    print("========================================================\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                continue

            # Flip horizontal agar seperti cermin
            img = cv2.flip(img, 1)
            h, w, c = img.shape

            # Frame galaxy disesuaikan dengan dimensi kamera aktif
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

                # 2. CEK GESTUR GANTI FILTER: Jempol & Kelingking Saling Mendekat pada 1 Tangan
                for hand_lms in results.multi_hand_landmarks:
                    thumb = hand_lms.landmark[4]
                    pinky = hand_lms.landmark[20]
                    tx, ty = int(thumb.x * w), int(thumb.y * h)
                    px, py = int(pinky.x * w), int(pinky.y * h)
                    if math.hypot(tx - px, ty - py) < 45:
                        change_filter = True

                # Trigger Pergantian Filter (Debounce)
                if change_filter:
                    if not gesture_triggered:
                        current_filter = (current_filter + 1) % len(FILTERS)
                        gesture_triggered = True
                        print(f"Gestur Terdeteksi! Pindah ke Filter: {FILTERS[current_filter]}")
                else:
                    gesture_triggered = False

                # Ambil koordinat Ujung Jempol (ID 4) dan Telunjuk (ID 8) untuk titik portal
                for hand_lms in results.multi_hand_landmarks:
                    # Gambar koneksi skeleton bawaan MediaPipe
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
                        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)

                # ========================================================
                # FILTER LAYAR PENUH (FULL FRAME MODE)
                # ========================================================
                if full_frame_mode:
                    img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

                # ========================================================
                # 4 POIN -> PORTAL MIRING (POLYGON) BETWEEN 2 HANDS
                # ========================================================
                elif len(pts_portal) == 4:
                    # Urutkan titik berdasarkan koordinat Y (Atas & Bawah)
                    pts_portal.sort(key=lambda p: p[1])
                    top_pts = pts_portal[:2]
                    bottom_pts = pts_portal[2:]

                    # Urutkan titik berdasarkan koordinat X (Kiri & Kanan)
                    top_pts.sort(key=lambda p: p[0])
                    bottom_pts.sort(key=lambda p: p[0])

                    poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)

                    # Dapatkan Bounding Box area Polygon
                    x, y, bw, bh = cv2.boundingRect(poly_pts)
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)

                    if bw > 0 and bh > 0:
                        roi = img[y:y+bh, x:x+bw].copy()
                        filtered_roi = apply_filter(roi, filter_name, x, y, None, frame_galaxy)

                        # Buat Mask Polygon untuk mengisolasi area portal
                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        poly_roi = poly_pts - [x, y]
                        cv2.fillPoly(mask, [poly_roi], 255)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        # Aplikasikan hasil filter hanya di dalam area polygon portal
                        img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)

                        # Gambar garis tepi portal
                        cv2.polylines(img, [poly_pts], True, (255, 255, 255), 3, cv2.LINE_AA)

                        # --- PARTIKEL GLOW DI TEPI PORTAL ---
                        for i in range(4):
                            p_start = poly_pts[i]
                            p_end = poly_pts[(i + 1) % 4]
                            for _ in range(5):
                                alpha = np.random.random()
                                px = int(p_start[0] * alpha + p_end[0] * (1 - alpha)) + np.random.randint(-12, 12)
                                py = int(p_start[1] * alpha + p_end[1] * (1 - alpha)) + np.random.randint(-12, 12)
                                cv2.circle(img, (px, py), np.random.randint(2, 5), (0, 255, 255), -1)

                        # Label Nama Portal
                        cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], max(30, top_pts[0][1] - 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # Jika hanya 1 tangan (2 poin), terapkan filter di sekeliling lingkaran jempol & telunjuk
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
                        cv2.circle(img, (cx, cy), radius, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(img, f"MINI PORTAL: {filter_name}", (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            elif full_frame_mode:
                img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

            # ========================================================
            # OVERLAY INFORMASI HUD PANEL ATAS
            # ========================================================
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.rectangle(img, (10, 10), (620, 110), (20, 20, 20), -1)
            cv2.rectangle(img, (10, 10), (620, 110), (0, 255, 255), 2)

            mode_type = "Full Frame" if full_frame_mode else "Portal (4 Points / 2 Hands)"
            cv2.putText(img, f"FPS: {int(fps)} | Mode: {mode_type}", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Filter Aktif: {filter_name} [{current_filter + 1}/{len(FILTERS)}]", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Ganti Filter: Touch Jempol+Kelingking | [TAB] Switch | [f] Toggle Full", (25, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Display Frame
            cv2.imshow(window_name, img)

            # Keyboard Input Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key in [9, 32]: # TAB atau SPACE
                current_filter = (current_filter + 1) % len(FILTERS)
                print(f"Pindah Filter ke: {FILTERS[current_filter]}")
            elif key == ord('f') or key == ord('F'):
                full_frame_mode = not full_frame_mode
                print(f"Toggle Full Frame Mode: {full_frame_mode}")
            elif ord('0') <= key <= ord('9'):
                idx_key = key - ord('0')
                if idx_key < len(FILTERS):
                    current_filter = idx_key
                    print(f"Pilih Filter Direct: {FILTERS[current_filter]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
