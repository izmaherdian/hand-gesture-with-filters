import sys
import os

# Prevent local './mediapipe' folder from shadowing the installed package
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath('.')]

import cv2
import mediapipe as mp
import time
import math
import numpy as np

# ── 11 RETROLENS Filters ──
FILTERS = [
    "MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA",
    "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"
]

# ── Minimal Color Palette (BGR) ──
C_ACCENT   = (255, 200, 60)    # Soft cyan-ish accent
C_ACTIVE   = (120, 80, 255)    # Warm violet for active pill
C_WHITE    = (245, 245, 245)
C_DIMMED   = (140, 140, 140)
C_PILL_BG  = (40, 38, 42)      # Near-black pill background
C_DOT_ON   = (80, 220, 120)    # Green dot for FPS
C_PORTAL_A = (255, 160, 80)    # Portal glow outer
C_PORTAL_B = (255, 220, 160)   # Portal glow inner
C_FINGER   = (255, 200, 100)   # Finger dot accent


def generate_galaxy_background(height=1080, width=1920):
    """Procedurally generate a Galaxy / Space starfield background."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = (30, 10, 40)

    # Small white stars
    for _ in range(800):
        sx, sy = np.random.randint(0, width), np.random.randint(0, height)
        bg[sy, sx] = (255, 255, 255)

    # Larger colored stars
    for _ in range(100):
        sx, sy = np.random.randint(0, width), np.random.randint(0, height)
        r = np.random.randint(2, 5)
        color = (np.random.randint(160, 255), np.random.randint(100, 255), 255)
        cv2.circle(bg, (sx, sy), r, color, -1)

    return bg


def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    """Apply one of 11 filters to the given Region of Interest."""
    if roi is None or roi.size == 0:
        return roi

    h_r, w_r = roi.shape[:2]

    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)

    elif filter_name == "BLUR":
        ksize = max(3, (min(h_r, w_r) // 10) | 1)
        return cv2.GaussianBlur(roi, (ksize, ksize), 0)

    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(roi, kernel), 0, 255).astype(np.uint8)

    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [219, 0, 189]
        return filtered

    elif filter_name == "PIXELATE":
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r // 12), max(1, h_r // 12)),
                               interpolation=cv2.INTER_LINEAR)
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
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = C_ACCENT
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)

    elif filter_name == "GALAXY" and frame_galaxy is not None:
        roi_galaxy = frame_galaxy[y:y+h_r, x:x+w_r]
        if roi_galaxy.shape[:2] == (h_r, w_r):
            if mask_person is not None:
                roi_mask = mask_person[y:y+h_r, x:x+w_r]
                filtered = roi.copy()
                filtered[roi_mask == 0] = roi_galaxy[roi_mask == 0]
                return filtered
            else:
                return cv2.addWeighted(roi, 0.45, roi_galaxy, 0.55, 0)

    return roi


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLEAN MINIMALIST HUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_minimal_hud(img, fps, current_filter_idx, full_frame_mode):
    """
    Draw a clean, minimal HUD that doesn't block the camera view:
      - Top-left corner: FPS + mode badge
      - Bottom center: Instagram-style filter carousel pills
      - Bottom-right corner: keyboard shortcuts hint
    All text uses FONT_HERSHEY_DUPLEX with larger sizes for HD clarity.
    """
    h, w = img.shape[:2]
    FONT = cv2.FONT_HERSHEY_DUPLEX  # Double-stroked = sharper on downscale

    # ── TOP-LEFT: FPS Badge ──
    overlay = img.copy()
    badge_w, badge_h = 220, 44
    bx, by = 20, 18
    cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h),
                  (20, 18, 22), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Green dot + FPS text
    cv2.circle(img, (bx + 16, by + badge_h // 2), 6, C_DOT_ON, -1, cv2.LINE_AA)
    cv2.putText(img, f"{int(fps)} FPS", (bx + 30, by + 30),
                FONT, 0.65, C_WHITE, 1, cv2.LINE_AA)

    # Mode label
    mode_txt = "FULL" if full_frame_mode else "PORTAL"
    cv2.putText(img, mode_txt, (bx + 125, by + 30),
                FONT, 0.55, C_DIMMED, 1, cv2.LINE_AA)

    # ── BOTTOM CENTER: Instagram-style Filter Carousel ──
    overlay2 = img.copy()

    pill_w = 130
    pill_h = 42
    pill_gap = 8
    total_carousel_w = len(FILTERS) * (pill_w + pill_gap) - pill_gap
    carousel_x = (w - total_carousel_w) // 2
    carousel_y = h - 80

    # Frosted glass bar behind carousel
    bar_pad = 18
    bar_x1 = max(0, carousel_x - bar_pad)
    bar_x2 = min(w, carousel_x + total_carousel_w + bar_pad)
    bar_y1 = carousel_y - 14
    bar_y2 = h - 12
    cv2.rectangle(overlay2, (bar_x1, bar_y1), (bar_x2, bar_y2),
                  (18, 16, 20), -1)
    cv2.addWeighted(overlay2, 0.50, img, 0.50, 0, img)

    # Top border line
    cv2.line(img, (bar_x1, bar_y1), (bar_x2, bar_y1),
             (70, 68, 75), 1, cv2.LINE_AA)

    for i, f_name in enumerate(FILTERS):
        px1 = carousel_x + i * (pill_w + pill_gap)
        py1 = carousel_y
        px2 = px1 + pill_w
        py2 = py1 + pill_h
        is_active = (i == current_filter_idx)

        if is_active:
            cv2.rectangle(img, (px1, py1), (px2, py2), C_ACTIVE, -1)
            cv2.rectangle(img, (px1 - 1, py1 - 1), (px2 + 1, py2 + 1),
                          (180, 140, 255), 2, cv2.LINE_AA)
            txt_col = C_WHITE
        else:
            cv2.rectangle(img, (px1, py1), (px2, py2), C_PILL_BG, -1)
            txt_col = C_DIMMED

        # Truncate long names
        label = f_name if len(f_name) <= 9 else f_name[:8] + "."

        # Center text in pill
        text_size = cv2.getTextSize(label, FONT, 0.55, 1)[0]
        tx = px1 + (pill_w - text_size[0]) // 2
        ty = py1 + (pill_h + text_size[1]) // 2

        cv2.putText(img, label, (tx, ty), FONT, 0.55, txt_col, 1, cv2.LINE_AA)

    # Active filter name displayed above the carousel
    active_name = FILTERS[current_filter_idx]
    name_size = cv2.getTextSize(active_name, FONT, 0.85, 2)[0]
    nx = (w - name_size[0]) // 2
    ny = carousel_y - 24
    cv2.putText(img, active_name, (nx, ny), FONT, 0.85, C_WHITE, 2, cv2.LINE_AA)

    # ── BOTTOM-RIGHT: Shortcut Hints ──
    hints = "[TAB] Next   [F] Mode   [Q] Quit"
    cv2.putText(img, hints, (w - 480, h - 16),
                FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0

    print(f"Opening camera index: {cam_index}...")
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened() and cam_index != 0:
        print(f"Camera {cam_index} unavailable, falling back to camera 0...")
        cam_index = 0
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No camera available.")
        return

    # ── Camera: 1920x1080 @ 30 FPS MJPG ──
    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # ── Resizable Window (default fits 1080p laptop) ──
    win = "RETROLENS"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    galaxy_bg = generate_galaxy_background(1080, 1920)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    current_filter = 0
    gesture_triggered = False
    full_frame_mode = False
    fps_display = 30.0

    # Smoothed FPS using exponential moving average
    fps_alpha = 0.15

    print()
    print("  RETROLENS — Hand Gesture Portal Filters")
    print("  Camera : 1920x1080 @ 30 FPS (120 deg wide)")
    print("  Controls: [TAB] Next filter  [F] Toggle mode  [Q] Quit")
    print()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        while cap.isOpened():
            t_start = time.time()

            ok, img = cap.read()
            if not ok:
                continue

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            frame_galaxy = galaxy_bg[:h, :w]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            filter_name = FILTERS[current_filter]
            pts_portal = []
            change_filter = False

            if results.multi_hand_landmarks:
                # ── Gesture: Two index fingertips touching ──
                if len(results.multi_hand_landmarks) >= 2:
                    idx0 = results.multi_hand_landmarks[0].landmark[8]
                    idx1 = results.multi_hand_landmarks[1].landmark[8]
                    pt0 = (int(idx0.x * w), int(idx0.y * h))
                    pt1 = (int(idx1.x * w), int(idx1.y * h))
                    if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 45:
                        change_filter = True

                # ── Gesture: Thumb + pinky touching ──
                for hand_lms in results.multi_hand_landmarks:
                    thumb = hand_lms.landmark[4]
                    pinky = hand_lms.landmark[20]
                    tx, ty = int(thumb.x * w), int(thumb.y * h)
                    px, py = int(pinky.x * w), int(pinky.y * h)
                    if math.hypot(tx - px, ty - py) < 45:
                        change_filter = True

                # Debounce
                if change_filter:
                    if not gesture_triggered:
                        current_filter = (current_filter + 1) % len(FILTERS)
                        gesture_triggered = True
                else:
                    gesture_triggered = False

                # ── Collect portal points + draw hand skeleton ──
                for hand_lms in results.multi_hand_landmarks:
                    # Draw clean hand skeleton
                    mp_draw.draw_landmarks(
                        img, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    for id_lm in [4, 8]:
                        cx = int(hand_lms.landmark[id_lm].x * w)
                        cy = int(hand_lms.landmark[id_lm].y * h)
                        pts_portal.append([cx, cy])

                        # Small, clean finger dots
                        cv2.circle(img, (cx, cy), 8, C_FINGER, 2, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), 3, C_WHITE, -1, cv2.LINE_AA)

                # ── MODE: Full Frame ──
                if full_frame_mode:
                    img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

                # ── MODE: 4-Point Portal (2 hands) ──
                elif len(pts_portal) == 4:
                    pts_portal.sort(key=lambda p: p[1])
                    top_pts = sorted(pts_portal[:2], key=lambda p: p[0])
                    bottom_pts = sorted(pts_portal[2:], key=lambda p: p[0])

                    poly_pts = np.array(
                        [top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]],
                        dtype=np.int32)

                    x, y, bw, bh = cv2.boundingRect(poly_pts)
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)

                    if bw > 0 and bh > 0:
                        roi = img[y:y+bh, x:x+bw].copy()
                        filtered_roi = apply_filter(roi, filter_name, x, y,
                                                    None, frame_galaxy)

                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_pts - [x, y]], 255)
                        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        img[y:y+bh, x:x+bw] = np.where(mask3 == 255,
                                                         filtered_roi, roi)

                        # Clean thin portal border (single line, not bulky)
                        cv2.polylines(img, [poly_pts], True, C_PORTAL_B, 2,
                                      cv2.LINE_AA)

                        # Subtle sparkles along edges
                        for i in range(4):
                            p1 = poly_pts[i]
                            p2 = poly_pts[(i + 1) % 4]
                            for _ in range(3):
                                a = np.random.random()
                                sx = int(p1[0]*a + p2[0]*(1-a)) + np.random.randint(-8, 8)
                                sy = int(p1[1]*a + p2[1]*(1-a)) + np.random.randint(-8, 8)
                                cv2.circle(img, (sx, sy), np.random.randint(1, 3),
                                           C_PORTAL_A, -1)

                # ── MODE: Mini Portal (1 hand) ──
                elif len(pts_portal) == 2 and not full_frame_mode:
                    mx = (pts_portal[0][0] + pts_portal[1][0]) // 2
                    my = (pts_portal[0][1] + pts_portal[1][1]) // 2
                    radius = int(math.hypot(
                        pts_portal[0][0] - pts_portal[1][0],
                        pts_portal[0][1] - pts_portal[1][1])) // 2 + 30

                    x1, y1 = max(0, mx - radius), max(0, my - radius)
                    x2, y2 = min(w, mx + radius), min(h, my + radius)
                    bw, bh = x2 - x1, y2 - y1

                    if bw > 0 and bh > 0:
                        roi = img[y1:y2, x1:x2].copy()
                        filtered_roi = apply_filter(roi, filter_name, x1, y1,
                                                    None, frame_galaxy)

                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        cv2.circle(mask, (mx - x1, my - y1), radius, 255, -1)
                        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        img[y1:y2, x1:x2] = np.where(mask3 == 255,
                                                       filtered_roi, roi)
                        cv2.circle(img, (mx, my), radius, C_PORTAL_B, 2,
                                   cv2.LINE_AA)

            elif full_frame_mode:
                img = apply_filter(img, filter_name, 0, 0, None, frame_galaxy)

            # ── 30 FPS Frame Pacing ──
            elapsed = time.time() - t_start
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)

            actual_fps = 1.0 / max(time.time() - t_start, 0.001)
            fps_display = fps_alpha * actual_fps + (1 - fps_alpha) * fps_display

            # ── Draw Clean HUD ──
            draw_minimal_hud(img, fps_display, current_filter, full_frame_mode)

            cv2.imshow(win, img)

            # ── Keyboard Controls ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key in [9, 32]:  # TAB or SPACE
                current_filter = (current_filter + 1) % len(FILTERS)
            elif key == ord('f') or key == ord('F'):
                full_frame_mode = not full_frame_mode
            elif ord('0') <= key <= ord('9'):
                idx = key - ord('0')
                if idx < len(FILTERS):
                    current_filter = idx

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
