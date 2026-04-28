#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys


def detect_colored_circles(camera_index=0):
    """
    Nhận diện vòng tròn rỗng màu đỏ, vàng, xanh dương từ camera.
    Tương thích Python 3.6 và OpenCV 3/4.
    """

    try:
        camera_index = int(camera_index)
    except Exception:
        camera_index = 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Không thể mở camera: {}".format(camera_index))
        return

    print("Đang mở camera: {}".format(camera_index))
    print("Bắt đầu phát hiện vòng tròn rỗng... Nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Không đọc được frame từ camera")
            break

        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2

        cv2.putText(
            frame,
            "TARGET",
            (center_x - 30, center_y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v_ch)
        hsv = cv2.merge([h_ch, s_ch, v_eq])

        scene_v_mean = float(np.mean(v_eq))
        if scene_v_mean > 200:
            adaptive_s_min = 100
            adaptive_v_min = 70
        elif scene_v_mean > 170:
            adaptive_s_min = 100
            adaptive_v_min = 60
        else:
            adaptive_s_min = 100
            adaptive_v_min = 50

        colors = {
            'RED': {
                'lower1': np.array([0, adaptive_s_min, adaptive_v_min]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, adaptive_s_min, adaptive_v_min]),
                'upper2': np.array([180, 255, 255]),
                'bgr': (0, 0, 255)
            },
            'YELLOW': {
                'lower': np.array([14, adaptive_s_min, adaptive_v_min]),
                'upper': np.array([40, 255, 255]),
                'bgr': (0, 255, 255)
            },
            'BLUE': {
                'lower': np.array([100, adaptive_s_min, adaptive_v_min]),
                'upper': np.array([125, 255, 255]),
                'bgr': (255, 0, 0)
            }
        }

        min_contour_area = 240
        min_radius = 10
        min_circularity = 0.35
        ring_fill_ratio_min = 0.05
        ring_fill_ratio_max = 0.70
        center_fill_ratio_max = 0.28
        min_hole_ratio = 0.08

        y_text = 30
        detected_any = False

        for color_name, color_info in colors.items():
            if color_name == 'RED':
                mask1 = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
                mask2 = cv2.inRange(hsv, color_info['lower2'], color_info['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])

            neutral_bright_mask = cv2.inRange(
                hsv,
                np.array([0, 0, 205]),
                np.array([180, 60, 255])
            )
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(neutral_bright_mask))

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            found = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if len(found) == 2:
                contours, hierarchy = found
            else:
                _, contours, hierarchy = found

            best_candidate = None
            best_area = 0.0

            if hierarchy is None:
                continue

            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue

                parent_idx = hierarchy[0][idx][3]
                if parent_idx != -1:
                    continue

                child_idx = hierarchy[0][idx][2]

                hole_area = 0.0
                cur_child = child_idx
                while cur_child != -1:
                    hole_area += cv2.contourArea(contours[cur_child])
                    cur_child = hierarchy[0][cur_child][0]

                hole_ratio = hole_area / area if area > 0 else 0.0
                if child_idx != -1 and hole_ratio < min_hole_ratio:
                    continue

                ring_area = area - hole_area
                if ring_area < min_contour_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                if circularity < min_circularity:
                    continue

                (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(contour)
                radius = int(radius_f)
                if radius < min_radius:
                    continue

                circle_area = np.pi * (radius ** 2)
                if circle_area <= 0:
                    continue

                fill_ratio = ring_area / circle_area
                if not (ring_fill_ratio_min <= fill_ratio <= ring_fill_ratio_max):
                    continue

                cx = int(cx_f)
                cy = int(cy_f)
                patch_half = max(4, int(radius * 0.25))
                x1 = max(0, cx - patch_half)
                y1 = max(0, cy - patch_half)
                x2 = min(mask.shape[1], cx + patch_half + 1)
                y2 = min(mask.shape[0], cy + patch_half + 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                center_patch = mask[y1:y2, x1:x2]
                center_fill_ratio = cv2.countNonZero(center_patch) / float(center_patch.size)
                if center_fill_ratio > center_fill_ratio_max:
                    continue

                if ring_area > best_area:
                    best_area = ring_area
                    best_candidate = (cx, cy, radius)

            if best_candidate is not None:
                circle_x, circle_y, radius = best_candidate
                detected_any = True

                cv2.circle(frame, (circle_x, circle_y), radius, color_info['bgr'], 2)
                cv2.circle(frame, (circle_x, circle_y), 5, color_info['bgr'], -1)
                cv2.line(
                    frame,
                    (circle_x, circle_y),
                    (center_x, center_y),
                    color_info['bgr'],
                    2,
                    cv2.LINE_AA
                )

                directions = []
                tolerance = 30

                if circle_y < center_y - tolerance:
                    directions.append("DOWN")
                elif circle_y > center_y + tolerance:
                    directions.append("UP")

                if circle_x < center_x - tolerance:
                    directions.append("LEFT")
                elif circle_x > center_x + tolerance:
                    directions.append("RIGHT")

                direction_str = "CENTER" if not directions else " + ".join(directions)

                print("[{}] X: {:3d}, Y: {:3d} | R: {:2d}px -> {}".format(
                    color_name, circle_x, circle_y, radius, direction_str
                ))

                info_text = "{}: {}".format(color_name, direction_str)
                cv2.putText(
                    frame,
                    info_text,
                    (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_info['bgr'],
                    2
                )
                y_text += 30

        if not detected_any:
            cv2.putText(
                frame,
                "Khong phat hien vong tron",
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2
            )

        cv2.imshow("Circle Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== CIRCLE DETECTION - PHAT HIEN VONG TRON ===\n")
    print("Chế độ: Camera mặc định -> 0")
    detect_colored_circles(0)