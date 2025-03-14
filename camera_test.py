import cv2
import time
from utils.helper import new_target_dimensions

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./videos/higher.mp4")
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_width / original_height

target_width = 224
target_height = 224


new_width, new_height = new_target_dimensions(cap, target_width, target_height)
prev_frame_time = 0
curr_frame_time = 0
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

prev_frame_time = 0
curr_frame_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_width, new_height),interpolation=cv2.INTER_AREA)
    print(
        f"Original size: {original_width}x{original_height}, New size: {new_width}x{new_height}"
    )
    if not ret:
        print("Không thể nhận dữ liệu từ camera")
        break

    curr_frame_time = time.time()
    fps = 1 / (curr_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_frame_time

    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
