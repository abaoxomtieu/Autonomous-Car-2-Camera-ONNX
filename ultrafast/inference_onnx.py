from .ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from config.setting import car_length_padding, per_len_lane, permission_rotate_angle
import cv2
import numpy as np
import math
import os
from loguru import logger

model_type = ModelType.TUSIMPLE

dirname = os.path.dirname(__file__)

model_path = os.path.join(dirname, "./models/tusimple_18_V1_fp32.onnx")
inference = UltrafastLaneDetector(model_path, model_type)
lane_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
height = 720
width = 1280

car_point_left = (car_length_padding, height)
car_point_right = (width - car_length_padding, height)
car_center_bottom = ((car_point_left[0] + car_point_right[0]) // 2, height)
car_center_top = (car_center_bottom[0], 0)


def draw_lanes(
    input_img,
    lanes_points,
    lanes_detected,
    left_points_90=None,
    right_points_90=None,
    draw_points=True,
    caculate=True,
):
    left_top = None
    right_top = None
    Have_lane = True
    visualization_img = cv2.resize(input_img, (1280, 720), interpolation=cv2.INTER_AREA)
    if lanes_detected[1] and lanes_detected[2] and caculate:
        lane_segment_img = input_img.copy()

        # Chuyển các điểm của lane trái và phải sang numpy array
        left_lane = np.array(lanes_points[1])
        right_lane = np.array(lanes_points[2])

        # Tính y_top và y_bottom của từng lane
        y_top_left = np.min(left_lane[:, 1])
        y_bottom_left = np.max(left_lane[:, 1])
        y_top_right = np.min(right_lane[:, 1])
        y_bottom_right = np.max(right_lane[:, 1])

        # Xác định vùng giao nhau của 2 lane theo trục y
        y_lane_top = max(y_top_left, y_top_right)
        y_lane_bottom = min(y_bottom_left, y_bottom_right)
        lane_length = y_lane_bottom - y_lane_top

        # Xác định ngưỡng y cho 90% chiều dài (phần gần camera)
        y_threshold = y_lane_bottom - per_len_lane * lane_length

        # Lọc các điểm của lane theo ngưỡng y (chỉ lấy phần gần camera)
        left_points_90 = [point for point in lanes_points[1] if point[1] >= y_threshold]
        right_points_90 = [
            point for point in lanes_points[2] if point[1] >= y_threshold
        ]
        # Tính tọa độ của cạnh trên và cạnh dưới cho lane trái
        if left_points_90:
            left_top = min(left_points_90, key=lambda p: p[1])  # Điểm có y nhỏ nhất
            # left_bottom = max(left_points_90, key=lambda p: p[1])  # Điểm có y lớn nhất
        # Tính tọa độ của cạnh trên và cạnh dưới cho lane phải
        if right_points_90:
            right_top = min(right_points_90, key=lambda p: p[1])
            # right_bottom = max(right_points_90, key=lambda p: p[1])

        # Nếu có đủ điểm từ cả hai lane, tiến hành vẽ
        if len(left_points_90) > 0 and len(right_points_90) > 0:
            pts = np.vstack(
                (np.array(left_points_90), np.flipud(np.array(right_points_90)))
            )
            cv2.fillPoly(lane_segment_img, pts=[pts], color=(255, 191, 0))
            visualization_img = cv2.addWeighted(
                visualization_img, 0.7, lane_segment_img, 0.3, 0
            )
        else:
            Have_lane = False

    if draw_points:
        if left_points_90 and right_points_90:
            if len(left_points_90) > 0 and len(right_points_90) > 0:
                lane_segment_img = input_img.copy()
                pts = np.vstack(
                    (np.array(left_points_90), np.flipud(np.array(right_points_90)))
                )
                cv2.fillPoly(lane_segment_img, pts=[pts], color=(255, 191, 0))
                visualization_img = cv2.addWeighted(
                    visualization_img, 0.7, lane_segment_img, 0.3, 0
                )
            else:
                Have_lane = False
        for lane_num, lane_points in enumerate(lanes_points):
            for lane_point in lane_points:
                cv2.circle(
                    visualization_img,
                    (lane_point[0], lane_point[1]),
                    3,
                    lane_colors[lane_num],
                    -1,
                )
    return (
        visualization_img,
        left_top,
        right_top,
        left_points_90,
        right_points_90,
        Have_lane,
    )


def process_output(
    frame,
    lanes_points=None,
    lanes_detected=None,
    left_top_cache=None,
    right_top_cache=None,
    left_points_90_cache=None,
    right_points_90_cache=None,
    paint=True,
    resize_img=True,
    calculate=False,
):
    direction = None
    Have_lane = True
    if lanes_points is None or lanes_detected is None:
        lanes_points, lanes_detected = inference.detect_lanes(frame)
    (
        visualization_img,
        left_top,
        right_top,
        left_points_90,
        right_points_90,
        Have_lane,
    ) = draw_lanes(
        frame,
        lanes_points,
        lanes_detected,
        left_points_90_cache,
        right_points_90_cache,
        draw_points=True,
        caculate=calculate,
    )
    if not calculate:
        left_top = left_top_cache
        right_top = right_top_cache

    if not Have_lane:
        logger.info("NO LANE")
        return visualization_img, direction, Have_lane, left_top, right_top, None, None
    if not left_top or not right_top:
        logger.info("No leftop, righTop")
        return visualization_img, direction, Have_lane, left_top, right_top, None, None
    if paint:
        car_points = [
            car_point_left,
            car_center_bottom,
            car_point_right,
            car_center_top,
        ]
        for pt in car_points:
            cv2.circle(visualization_img, pt, 10, (50, 100, 255), -1)

    if left_top and right_top:
        top_center = (
            (left_top[0] + right_top[0]) // 2,
            (left_top[1] + right_top[1]) // 2,
        )
        if paint:
            for pt in [left_top, right_top, top_center]:
                cv2.circle(
                    visualization_img,
                    pt,
                    5 if pt != top_center else 7,
                    (0, 255, 255) if pt != top_center else (0, 0, 255),
                    -1,
                )

        point_control_left, point_control_right = (left_top[0], height), (
            right_top[0],
            height,
        )

        if paint:
            for pt in [point_control_left, point_control_right]:
                cv2.circle(visualization_img, pt, 10, (100, 255, 100), -1)

        dx, dy = (
            top_center[0] - car_center_bottom[0],
            car_center_bottom[1] - top_center[1],
        )
        angle_deg = math.degrees(math.atan2(dx, dy))
        direction = (
            "LEFT"
            if angle_deg < -(permission_rotate_angle)
            else "RIGHT" if angle_deg > (permission_rotate_angle) else "STRAIGHT"
        )
        if paint:
            cv2.putText(
                visualization_img,
                f"{direction} ({angle_deg:.2f} deg)",
                (15, 500),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                5,
            )

    if resize_img:
        visualization_img = cv2.resize(
            visualization_img,
            (visualization_img.shape[1] // 2, visualization_img.shape[0] // 2),
        )

    return (
        visualization_img,
        direction,
        Have_lane,
        left_top,
        right_top,
        left_points_90,
        right_points_90,
    )
