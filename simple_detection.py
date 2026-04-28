#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import sys

def _resolve_video_source(source=None):
    if source is not None:
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    default_video = os.path.join(os.path.dirname(__file__), "IMG_1404.MOV")
    if os.path.exists(default_video):
        return default_video

    return 0


def detect_colored_circles(source=None):
    """
    Nhận diện vòng tròn rỗng màu đỏ, vàng, xanh dương
    Dùng Color Filter + kiểm tra hình dạng vành tròn (rỗng ruột)
    In ra hướng cần di chuyển để tâm trùng
    
    Args:
        source: ID camera hoặc đường dẫn video
    """

    source = _resolve_video_source(source)

    # Mở camera hoặc video file
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Không thể mở nguồn video: {source}")
        return
    
    print(f"Đang mở nguồn: {source}")
    print("Bắt đầu phát hiện vòng tròn rỗng... Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lấy kích thước frame
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        

        cv2.putText(frame, "TARGET", (center_x - 30, center_y - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Tiền xử lý để ổn định nhận diện khi nền quá sáng
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
        
        # Định nghĩa phạm vi màu (loại bỏ màu nhạt - Saturation cao)
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

        # Ngưỡng kiểm tra hình vành tròn rỗng
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
            # Tạo mask cho màu
            if color_name == 'RED':
                mask1 = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
                mask2 = cv2.inRange(hsv, color_info['lower2'], color_info['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            # Loại vùng sáng trung tính (nền trắng/chói) để giảm nhiễu
            neutral_bright_mask = cv2.inRange(
                hsv,
                np.array([0, 0, 205]),
                np.array([180, 60, 255])
            )
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(neutral_bright_mask))

            # Làm sạch mask nhưng giữ lỗ ở giữa để phân biệt vòng rỗng
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Tìm contour + hierarchy để bắt buộc có lỗ bên trong (vòng rỗng)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            best_candidate = None
            best_area = 0.0

            if hierarchy is None:
                continue

            # hierarchy[i] = [next, prev, first_child, parent]
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_contour_area:
                    continue

                # Chỉ lấy contour ngoài
                parent_idx = hierarchy[0][idx][3]
                if parent_idx != -1:
                    continue

                child_idx = hierarchy[0][idx][2]

                # Tính tổng diện tích lỗ con để đảm bảo là vành tròn rỗng
                hole_area = 0.0
                cur_child = child_idx
                while cur_child != -1:
                    hole_area += cv2.contourArea(contours[cur_child])
                    cur_child = hierarchy[0][cur_child][0]  # next child

                hole_ratio = hole_area / area if area > 0 else 0.0
                if child_idx != -1 and hole_ratio < min_hole_ratio:
                    continue

                # Diện tích vành thực tế = diện tích ngoài - diện tích lỗ
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

                # Vòng tròn rỗng phải có mật độ màu thấp ở vùng tâm
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

                # Vẽ vòng tròn phát hiện được
                cv2.circle(frame, (circle_x, circle_y), radius, color_info['bgr'], 2)
                cv2.circle(frame, (circle_x, circle_y), 5, color_info['bgr'], -1)

                # Vẽ đường kết nối
                cv2.line(frame, (circle_x, circle_y), (center_x, center_y),
                        color_info['bgr'], 2, cv2.LINE_AA)

                # Xác định hướng
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

                # In thông tin
                info_text = f"{color_name}: {direction_str}"
                print(f"[{color_name}] X: {circle_x:3d}, Y: {circle_y:3d} | R: {radius:2d}px -> {direction_str}")

                # Hiển thị lên màn hình
                cv2.putText(frame, info_text, (10, y_text),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_info['bgr'], 2)
                y_text += 30
        
        # Hiển thị thông báo nếu không phát hiện
        if not detected_any:
            cv2.putText(frame, "Khong phat hien vong tron", (10, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Hiển thị frame
        cv2.imshow("Circle Detection - Press 'q' to quit", frame)
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== CIRCLE DETECTION - PHÁT HIỆN VÒNG TRÒN ===\n")
    if len(sys.argv) > 1:
        print(f"Chế độ: Video/Camera từ đối số -> {sys.argv[1]}")
        detect_colored_circles(sys.argv[1])
    else:
        default_source = _resolve_video_source()
        print(f"Chế độ: Nguồn mặc định -> {default_source}")
        detect_colored_circles(default_source)

