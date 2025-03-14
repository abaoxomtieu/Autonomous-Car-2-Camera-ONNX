import cv2

def check_previous_directions(
    previous, value="STRAIGHT", number_of_last_frames=30, threshold=0.8
):
    straight_ratio = 0
    if len(previous) < 5:
        return False, straight_ratio

    last_samples = previous[-number_of_last_frames:]

    straight_count = sum(1 for direction in last_samples if direction == value)
    straight_ratio = straight_count / number_of_last_frames
    return straight_ratio >= threshold, straight_ratio
def visualize_frames(
    origin_capture=(None, None),
    dir=(None, None),
    dimenstion=(640, 480),
    ratio=(None, None),
    final_decision=None,
    fps=None,
):
    frame, frame_ = origin_capture
    direction, direction_ = dir
    dir_ratio, dir_ratio_ = ratio

    target_width, target_height = dimenstion

    frame_resized = cv2.resize(frame, (target_width, target_height))
    frame_resized_ = cv2.resize(frame_, (target_width, target_height))

    stacked_frames = cv2.hconcat([frame_resized, frame_resized_])

    # Add text overlays
    cv2.putText(
        stacked_frames,
        f"Upper Direction: {direction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # Red color
        2,
    )

    cv2.putText(
        stacked_frames,
        f"Lower Direction: {direction_}",
        (target_width + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),  # Red color
        2,
    )

    cv2.putText(
        stacked_frames,
        f"FPS: {fps:.2f}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green color
        2,
    )
    if final_decision is not None:
        # Calculate text size to center it
        text_size = cv2.getTextSize(final_decision, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]

        # Calculate center position
        text_x = (stacked_frames.shape[1] - text_size[0]) // 2
        text_y = stacked_frames.shape[0] - 20  # 20 pixels from bottom

        # Draw text with background for better visibility
        cv2.putText(
            stacked_frames,
            final_decision,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),  # Yellow color
            2,
        )
    if dir_ratio is not None:
        cv2.putText(
            stacked_frames,
            f"{direction} Ratio: {dir_ratio:.2f}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Cyan color
            2,
        )
    if dir_ratio_ is not None:
        cv2.putText(
            stacked_frames,
            f"{direction_} Ratio: {dir_ratio_:.2f}",
            (target_width + 10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Cyan color
            2,
        )

    return stacked_frames


def new_target_dimensions(cap: cv2.VideoCapture, target_width: int, target_height: int):
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    if target_width / target_height > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    return new_width, new_height
