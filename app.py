import cv2
import time
from utils.helper import (
    visualize_frames,
    new_target_dimensions,
    check_previous_directions,
)
from ultrafast.inference_onnx import process_output, inference
from classification.inference_onnx import inference_

# Open video sources
cap = cv2.VideoCapture("./videos/test_video.mp4")
cap_ = cv2.VideoCapture("./videos/lower.mp4")

real_car = False
if real_car:
    cap = cv2.VideoCapture(0)
    cap_ = cv2.VideoCapture(0)
new_width, new_height = new_target_dimensions(cap, 1280, 720)
new_width_, new_height_ = new_target_dimensions(cap, 224, 224)

# Initialize FPS calculation variables
frame_count = 0
start_time = time.time()
fps = 0
lanes_points = None
lanes_detected = None
visualization_img_cache = None


frame_infer = 0
skip_frames = 10

# model_
previous_ = []
temp_direction_ = 0
straight_ratio_ = None


# model
previous = []
temp_direction = 0
straight_ratio = None
final_decision = None
left_top_cache = None
right_top_cache = None
left_points_90_cache = None
right_points_90_cache = None
Have_lane = True
previous_direction = None

while cap.isOpened() and cap_.isOpened():
    frame_count += 1
    frame_infer += 1

    ret1, frame = cap.read()
    ret2, frame_ = cap_.read()
    show_frame = frame.copy()
    show_frame_ = frame_.copy()
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    frame_ = cv2.resize(frame_, (new_width_, new_height_))
    # time.sleep(1)
    if not ret1 or not ret2:
        print("End of one or both videos")
        break
    direction_, _ = inference_(frame_)
    # direction_ = "STRAIGHT"
    previous_.append(direction_)
    should_process_, straight_ratio_ = check_previous_directions(previous_, direction_)
    if should_process_:
        final_decision = direction_
    if frame_infer % skip_frames == 0 and should_process_ and direction_ == "STRAIGHT":
        lanes_points, lanes_detected = inference.detect_lanes(frame)
        (
            visualization_img,
            direction,
            Have_lane,
            left_top,
            right_top,
            left_points_90,
            right_points_90,
        ) = process_output(
            frame,
            lanes_points=lanes_points,
            left_top_cache=left_top_cache,
            right_top_cache=right_top_cache,
            left_points_90_cache=left_points_90_cache,
            right_points_90_cache=right_points_90_cache,
            lanes_detected=lanes_detected,
            calculate=True,
        )
        previous.append(direction)
        if direction:
            should_process, straight_ratio = check_previous_directions(
                previous, direction, 10, 0.7
            )
            if should_process:
                final_decision = direction
                previous_direction = direction

            left_top_cache = left_top
            right_top_cache = right_top
            left_points_90_cache = left_points_90
            right_points_90_cache = right_points_90
    else:
        visualization_img, direction, Have_lane, _, _, _, _ = process_output(
            frame,
            lanes_points=lanes_points,
            left_top_cache=left_top_cache,
            right_top_cache=right_top_cache,
            left_points_90_cache=left_points_90_cache,
            right_points_90_cache=right_points_90_cache,
            lanes_detected=lanes_detected,
            calculate=False,
        )
        if direction_ == "STRAIGHT" and should_process_:
            final_decision = previous_direction

    if visualization_img is None:
        print("failed")
        break

    elapsed_time = time.time() - start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time

        # Reset counters
        frame_count = 0
        start_time = time.time()

    stacked_frames = visualize_frames(
        (visualization_img, show_frame_),
        (direction, direction_),
        (640, 480),
        (straight_ratio, straight_ratio_),
        final_decision=final_decision,
        fps=fps,
    )

    cv2.imshow("Dual Camera View", stacked_frames)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cap_.release()
cv2.destroyAllWindows()
