import onnxruntime as ort
import cv2
import numpy as np
from numpy.typing import NDArray
import os


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Try CPU provider first
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        return session

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


dirname = os.path.dirname(__file__)
session = load_model(os.path.join(dirname, "./models/model_16.onnx"))


def prepare_input(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float16)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    img = (img / 255.0 - mean) / std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float16)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


classes = ["LEFT", "RIGHT", "STRAIGHT"]


def inference(image: NDArray[np.uint8]) -> tuple[int, float, NDArray[np.float16]]:
    """
    Run inference on an image and return class prediction with probabilities.

    Args:
        session: ONNX runtime session
        image: Input image in BGR format

    Returns:
        tuple containing:
        - predicted class index (int)
        - confidence score (float)
        - probability distribution (numpy array)
    """
    input_tensor = prepare_input(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]
    probabilities = softmax(output[0])
    predicted_class = classes[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    return predicted_class, confidence


def process_video(
    video_path: str, session, output_path: str = None, display: bool = True
):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    classes = ["left", "right", "straight"]

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            max_index, confidence, probs = inference(session, frame)
            text = f"{classes[max_index]}: {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if display:
                cv2.imshow("Video Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(frame)

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def inference_(image):
    predicted_class, probabilities = inference(image)
    return predicted_class, probabilities
