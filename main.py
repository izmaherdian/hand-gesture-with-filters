import sys
import os

# Abaikan folder lokal './mediapipe' agar Python menggunakan pustaka site-packages resmi
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath('.')]

import cv2
import mediapipe as mp
import math
import time

def main():
    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Ambil index kamera dari argumen terminal jika ada, misal: python main.py 1 (1 untuk kamera USB)
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0

    print(f"Mencoba membuka kamera index: {cam_index}...")
    cap = cv2.VideoCapture(cam_index)

    # Fallback ke kamera default (0) jika kamera USB yang diminta tidak dapat dibuka
    if not cap.isOpened() and cam_index != 0:
        print(f"Kamera index {cam_index} tidak dapat dibuka, mengalihkan ke kamera default (0)...")
        cam_index = 0
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera/webcam apapun.")
        return

    # Set format MJPEG untuk performa FPS tinggi (30/60 FPS) pada kamera USB 1080p
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # Set resolusi kamera ke Full HD 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    prev_time = 0

    print("\n========================================================")
    print(" 🚀 Hand Gesture Filter Started!")
    print(" 📷 Mode Kamera: Full HD 1920x1080 (Wide Angle 120° Optimized)")
    print(" - Dekatkan Ujung Jempol & Telunjuk untuk efek Pinch Filter")
    print(" - Gerakkan Telunjuk untuk memindahkan Halo Ring")
    print(" - Tekan 'q' atau 'ESC' pada jendela kamera untuk keluar")
    print("========================================================\n")

    # Nilai confidence 0.5 dioptimalkan untuk Wide Angle 120° agar tangan kecil/jauh tetap terdeteksi tajam
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Mengabaikan frame kosong dari kamera.")
                continue

            # Flip horizontal agar seperti cermin
            image = cv2.flip(image, 1)
            h, w, c = image.shape

            # Konversi BGR (OpenCV) ke RGB (MediaPipe)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            gesture_text = "Status: Menunggu Tangan..."
            filter_mode = "Normal"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Gambar landmark & koneksi bawaan MediaPipe
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Ambil koordinat Ujung Jempol (ID 4) & Telunjuk (ID 8)
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]

                    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    index_pos = (int(index_tip.x * w), int(index_tip.y * h))

                    # Hitung jarak piksel antara jempol dan telunjuk
                    distance = math.hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])

                    # Pusat antara jempol & telunjuk
                    mid_x = (thumb_pos[0] + index_pos[0]) // 2
                    mid_y = (thumb_pos[1] + index_pos[1]) // 2

                    # --- FILTER 1: EFEK HALO RING (Ujung Telunjuk) ---
                    cv2.circle(image, index_pos, 25, (255, 0, 128), 2, cv2.LINE_AA)
                    cv2.circle(image, index_pos, 15, (0, 255, 255), -1, cv2.LINE_AA)

                    # --- FILTER 2: GESTUR PINCH / MENJEPIT ---
                    pinch_threshold = int(w * 0.04) # Skala dinamis ~76px pada 1080p untuk Wide Angle
                    if distance < pinch_threshold:
                        gesture_text = "Gestur: PINCH (Menjepit)! Efek Energi Aktif"
                        filter_mode = "Cyan Glow"

                        # Gambar efek garis energi penghubung & aura bercahaya
                        cv2.line(image, thumb_pos, index_pos, (255, 255, 0), 4, cv2.LINE_AA)
                        cv2.circle(image, (mid_x, mid_y), 35, (255, 255, 0), 4, cv2.LINE_AA)
                        cv2.circle(image, (mid_x, mid_y), 20, (0, 255, 255), -1, cv2.LINE_AA)

                        # Efek tint warna latar layar (Cyberpunk / Neon Filter)
                        overlay = image.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (255, 200, 0), -1)
                        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
                    else:
                        gesture_text = f"Gestur: Terbuka | Jarak Jari: {int(distance)}px"
                        cv2.line(image, thumb_pos, index_pos, (0, 255, 0), 2, cv2.LINE_AA)

            # Hitung & Tampilkan FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Tampilan UI Panel Atas (Header)
            cv2.rectangle(image, (10, 10), (520, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (520, 100), (0, 255, 255), 2)

            cv2.putText(image, f"FPS: {int(fps)}", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, gesture_text, (25, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"Filter: {filter_mode}", (400, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Tampilkan window kamera
            cv2.imshow("Hand Gesture Filter - Level 1 Demo", image)

            # Tekan 'q' atau ESC untuk keluar
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
