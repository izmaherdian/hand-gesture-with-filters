import sys
import os

# Abaikan folder lokal './mediapipe' agar Python menggunakan pustaka site-packages resmi
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath('.')]

import cv2
import mediapipe as mp
import numpy as np
import math
import time

def count_fingers(landmarks, hand_label="Right"):
    """Menghitung jumlah jari yang sedang terbuka (0 - 5)"""
    fingers = []

    # Jempol (Thumb) - ID 4 vs ID 3 (x-axis tergantung tangan kiri/kanan)
    if hand_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # 4 Jari Lainnya (Telunjuk ID 8, Tengah ID 12, Manis ID 16, Kelingking ID 20)
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        # Dibandingkan dengan persendian di bawahnya (tip_id - 2)
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return sum(fingers), fingers

def get_gesture_name(fingers, is_pinch):
    """Menentukan nama gestur berdasarkan jari yang terbuka"""
    if is_pinch:
        return "PINCH (Menjepit)"
    total = sum(fingers)
    if total == 0:
        return "FIST (Mengepal)"
    elif total == 5:
        return "OPEN PALM (Tangan Terbuka)"
    elif fingers == [0, 1, 0, 0, 0]:
        return "POINTING (Menunjuk)"
    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE / VICTORY (V)"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"
    elif fingers == [0, 1, 0, 0, 1]:
        return "ROCK ON 🤘"
    return f"{total} Jari Terbuka"

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

    # Set format MJPEG, resolusi 1920x1080, dan 30 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Inisialisasi Window Resizable agar pas di layar laptop
    window_name = "Hand Gesture Multi-Filters System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Mode Filter:
    # 0: Normal + Hand Landmarks
    # 1: Air Canvas (Melukis di Udara)
    # 2: Interactive Drag & Drop Object
    # 3: Cyberpunk & Energy Lightning
    # 4: AR Magic Portal Rings
    # 5: Cartoon & Edge Detection
    filter_mode = 0
    filter_names = [
        "0: Normal Tracking",
        "1: Air Canvas (Melukis)",
        "2: Drag & Drop Object",
        "3: Cyberpunk & Energy",
        "4: AR Magic Ring",
        "5: Cartoon & Edge Filter"
    ]

    # --- VARIABEL UNTUK AIR CANVAS (MODE 1) ---
    canvas = None
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)] # Biru, Hijau, Merah, Kuning, Penghapus
    color_names = ["Biru", "Hijau", "Merah", "Kuning", "Eraser"]
    current_color_idx = 0
    brush_thickness = 8
    eraser_thickness = 40
    xp, yp = 0, 0

    # --- VARIABEL UNTUK DRAG & DROP (MODE 2) ---
    rect_pos = [400, 300] # [x, y]
    rect_size = [200, 200]
    is_dragging = False

    prev_time = 0

    print("\n========================================================")
    print(" 🚀 Hand Gesture Multi-Filter System Ready!")
    print(" 📷 Resolusi: Full HD 1920x1080 @ 30 FPS (120° Wide Angle)")
    print(" ------------------------------------------------------")
    print(" ⌨️  KONTROL FILTER:")
    print("   [0] Normal Tracking Mode")
    print("   [1] Air Canvas (Melukis di Udara)")
    print("   [2] Interactive Drag & Drop Box")
    print("   [3] Cyberpunk & Energy Aura Filter")
    print("   [4] AR Magic Portal Rings")
    print("   [5] Cartoon & Canny Edge Filter")
    print("   [c] Clear Canvas (Hapus Lukisan di Mode 1)")
    print("   [q / ESC] Keluar dari Aplikasi")
    print("========================================================\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip horizontal agar seperti cermin
            image = cv2.flip(image, 1)
            h, w, c = image.shape

            # Inisialisasi kanvas lukis jika belum ada
            if canvas is None or canvas.shape != image.shape:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Konversi warna BGR ke RGB untuk MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            gesture_text = "Menunggu Tangan..."

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label # "Right" atau "Left"
                    lm = hand_landmarks.landmark

                    # Hitung jari & gestur
                    total_fingers, finger_state = count_fingers(lm, hand_label)
                    
                    # Koordinat Ujung Jempol (ID 4) & Telunjuk (ID 8) & Tengah (ID 12)
                    thumb_pos = (int(lm[4].x * w), int(lm[4].y * h))
                    index_pos = (int(lm[8].x * w), int(lm[8].y * h))
                    middle_pos = (int(lm[12].x * w), int(lm[12].y * h))

                    distance_pinch = math.hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])
                    pinch_threshold = int(w * 0.04)
                    is_pinch = distance_pinch < pinch_threshold

                    gesture_text = f"[{hand_label}] {get_gesture_name(finger_state, is_pinch)}"

                    # --- GAMBAR LANDMARK HAND (Semua Mode kecuali Cartoon) ---
                    if filter_mode != 5:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                    # ========================================================
                    # MODE 1: AIR CANVAS (MELUKIS DI UDARA)
                    # ========================================================
                    if filter_mode == 1:
                        # Gestur 2 Jari Terbuka (Telunjuk + Tengah) = SELECTION MODE (Pilih Warna)
                        if finger_state[1] == 1 and finger_state[2] == 1 and finger_state[3] == 0:
                            xp, yp = 0, 0
                            cv2.rectangle(image, (index_pos[0]-15, index_pos[1]-15), 
                                          (middle_pos[0]+15, middle_pos[1]+15), (255, 255, 255), 2)
                            cv2.putText(image, "MODE: PILIH WARNA", (index_pos[0]-40, index_pos[1]-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Cek sentuhan di Palette Warna (Header)
                            if index_pos[1] < 120:
                                for idx in range(len(colors)):
                                    btn_x1 = 600 + idx * 110
                                    btn_x2 = btn_x1 + 100
                                    if btn_x1 < index_pos[0] < btn_x2:
                                        current_color_idx = idx

                        # Gestur 1 Jari Telunjuk Terbuka = DRAWING MODE (Menggambar)
                        elif finger_state[1] == 1 and finger_state[2] == 0:
                            cv2.circle(image, index_pos, 12, colors[current_color_idx], cv2.FILLED)
                            
                            if xp == 0 and yp == 0:
                                xp, yp = index_pos

                            thick = eraser_thickness if current_color_idx == 4 else brush_thickness
                            cv2.line(canvas, (xp, yp), index_pos, colors[current_color_idx], thick)
                            cv2.line(image, (xp, yp), index_pos, colors[current_color_idx], thick)
                            xp, yp = index_pos
                        else:
                            xp, yp = 0, 0

                    # ========================================================
                    # MODE 2: INTERACTIVE DRAG & DROP OBJECT
                    # ========================================================
                    elif filter_mode == 2:
                        rx, ry = rect_pos
                        rw, rh = rect_size

                        # Cek apakah Pinch di dalam area Kotak
                        if is_pinch and (rx < thumb_pos[0] < rx + rw) and (ry < thumb_pos[1] < ry + rh):
                            is_dragging = True
                            rect_pos[0] = thumb_pos[0] - rw // 2
                            rect_pos[1] = thumb_pos[1] - rh // 2
                        elif not is_pinch:
                            is_dragging = False

                    # ========================================================
                    # MODE 3: CYBERPUNK & ENERGY LIGHTNING
                    # ========================================================
                    elif filter_mode == 3:
                        # Garis petir antara setiap jari dan pergelangan
                        wrist_pos = (int(lm[0].x * w), int(lm[0].y * h))
                        for tip_id in [4, 8, 12, 16, 20]:
                            tip_pos = (int(lm[tip_id].x * w), int(lm[tip_id].y * h))
                            cv2.line(image, wrist_pos, tip_pos, (255, 255, 0), 2, cv2.LINE_AA)
                            cv2.circle(image, tip_pos, 15, (0, 255, 255), -1, cv2.LINE_AA)

                        if is_pinch:
                            mid_x = (thumb_pos[0] + index_pos[0]) // 2
                            mid_y = (thumb_pos[1] + index_pos[1]) // 2
                            cv2.circle(image, (mid_x, mid_y), 45, (0, 255, 255), 4, cv2.LINE_AA)
                            cv2.circle(image, (mid_x, mid_y), 25, (255, 255, 0), -1, cv2.LINE_AA)

                    # ========================================================
                    # MODE 4: AR MAGIC PORTAL RINGS
                    # ========================================================
                    elif filter_mode == 4:
                        wrist_pos = (int(lm[0].x * w), int(lm[0].y * h))
                        angle = int((time.time() * 100) % 360)

                        # Ring konsentris yang berputar di pergelangan
                        cv2.ellipse(image, wrist_pos, (60, 25), angle, 0, 360, (0, 255, 255), 3, cv2.LINE_AA)
                        cv2.ellipse(image, wrist_pos, (80, 35), -angle, 0, 360, (255, 0, 255), 2, cv2.LINE_AA)

                        # Aura di ujung jari
                        cv2.circle(image, index_pos, 20, (0, 200, 255), 2, cv2.LINE_AA)
                        cv2.circle(image, index_pos, 8, (255, 255, 255), -1, cv2.LINE_AA)

            # ========================================================
            # RENDER FILTER SPESIFIK (TAMPILAN MONITOR)
            # ========================================================

            # Render Canvas di Mode 1
            if filter_mode == 1:
                # Gabungkan lukisan canvas ke image kamera
                img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                image = cv2.bitwise_and(image, img_inv)
                image = cv2.bitwise_or(image, canvas)

                # Palette Warna Bar (UI Header atas)
                cv2.rectangle(image, (580, 10), (1150, 100), (40, 40, 40), -1)
                cv2.rectangle(image, (580, 10), (1150, 100), (255, 255, 255), 2)
                for idx, col in enumerate(colors):
                    bx = 600 + idx * 110
                    cv2.rectangle(image, (bx, 20), (bx + 90, 80), col, -1)
                    if idx == current_color_idx:
                        cv2.rectangle(image, (bx-4, 16), (bx + 94, 84), (255, 255, 255), 3)
                    cv2.putText(image, color_names[idx], (bx + 10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Render Drag & Drop Box di Mode 2
            elif filter_mode == 2:
                rx, ry = rect_pos
                rw, rh = rect_size
                box_color = (0, 255, 0) if is_dragging else (255, 100, 0)

                # Gambar kotak transparan bercahaya
                overlay = image.copy()
                cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), box_color, -1)
                cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), box_color, 4)

                status_box = "HOLD & DRAG WITH PINCH" if is_dragging else "PINCH TO DRAG BOX"
                cv2.putText(image, status_box, (rx + 10, ry + rh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Render Cyberpunk Background Overlay di Mode 3
            elif filter_mode == 3:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (150, 0, 150), -1)
                cv2.addWeighted(overlay, 0.12, image, 0.88, 0, image)

            # Render Cartoon & Edge Filter di Mode 5
            elif filter_mode == 5:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                              cv2.THRESH_BINARY, 9, 9)
                color_img = cv2.bilateralFilter(image, 9, 250, 250)
                image = cv2.bitwise_and(color_img, color_img, mask=edges)

            # ========================================================
            # UI PANEL ATAS (HEADER HUD & SELEKSI FILTER)
            # ========================================================
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Panel Utama Infobar
            cv2.rectangle(image, (10, 10), (550, 110), (20, 20, 20), -1)
            cv2.rectangle(image, (10, 10), (550, 110), (0, 255, 255), 2)

            cv2.putText(image, f"FPS: {int(fps)} | Mode: {filter_names[filter_mode]}", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Gestur: {gesture_text}", (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Tekan 0-5 utk Ganti Filter | [c] Clear Canvas | [q] Quit", (25, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Tampilkan window kamera
            cv2.imshow(window_name, image)

            # --- INPUT KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif ord('0') <= key <= ord('5'):
                filter_mode = key - ord('0')
                print(f"Mengubah Mode Filter ke: {filter_names[filter_mode]}")
            elif key == ord('c') or key == ord('C'):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                print("Kanvas lukis telah dibersihkan!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
