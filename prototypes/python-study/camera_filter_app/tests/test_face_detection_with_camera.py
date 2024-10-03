import math
import sys
from typing import Tuple, Union

import cv2
import numpy as np
from camera_filter_app.config.paths import AssetPaths
from mediapipe.python.solutions import drawing_utils, face_detection, face_mesh

# MediaPipe 초기화
mp_face_detection = face_detection
mp_face_mesh = face_mesh
mp_drawing = drawing_utils

TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    def is_valid_normalized_value(value: float) -> bool:
        return 0 <= value <= 1

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result:
        # Draw bounding box
        bbox = detection.location_data.relative_bounding_box
        start_point = _normalized_to_pixel_coordinates(
            bbox.xmin, bbox.ymin, width, height
        )
        end_point = _normalized_to_pixel_coordinates(
            bbox.xmin + bbox.width, bbox.ymin + bbox.height, width, height
        )
        if start_point and end_point:
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.location_data.relative_keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            if keypoint_px:
                cv2.circle(annotated_image, keypoint_px, 5, (0, 255, 0), -1)

    return annotated_image


def draw_landmarks_with_x(image, face_landmarks, landmark_indices):
    """Draws 'X' on the specified landmarks."""
    height, width, _ = image.shape
    for idx in landmark_indices:
        landmark = face_landmarks.landmark[idx]
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        if landmark_px:
            # Draw 'X'
            cv2.line(
                image,
                (landmark_px[0] - 5, landmark_px[1] - 5),
                (landmark_px[0] + 5, landmark_px[1] + 5),
                (0, 0, 255),
                1,
            )
            cv2.line(
                image,
                (landmark_px[0] - 5, landmark_px[1] + 5),
                (landmark_px[0] + 5, landmark_px[1] - 5),
                (0, 0, 255),
                1,
            )

    return image


def overlay_rabbit_ears(frame: cv2.Mat, landmarks: list) -> None:
    """
    This function overlays a rabbit ears image onto a specified frame using the landmarks provided by MediaPipe's Face Mesh.
    The algorithm calculates the position where the rabbit ears should be placed, crops the ears image if necessary, and then
    blends it into the frame using alpha blending for a smooth overlay effect.

    Args:
        frame (cv2.Mat): The input frame/image where the rabbit ears will be overlaid.
        landmarks (List[NormalizedLandmark]): The facial landmarks provided by MediaPipe's Face Mesh, used to determine the position of the rabbit ears.

    Returns:
        None: The function modifies the input frame directly to include the rabbit ears overlay.
    """
    ### Title: initialize
    ## Load the rabbit ears image with its alpha channel (transparency).
    rabbit_ear_img = cv2.imread(AssetPaths.FACE_RABBIT_EARS_PATH, cv2.IMREAD_UNCHANGED)
    if rabbit_ear_img is None:
        # If the image cannot be loaded, raise an error indicating the file is missing.
        raise FileNotFoundError("rabbit_ear2.png file not found.")

    # Get the dimensions of the input frame (height, width).
    h, w, _ = frame.shape

    ### Title: set rabbit ears in the center of frame

    # The index of the forehead landmark from the MediaPipe Face Mesh.
    forehead_index = 10
    # Retrieve the landmark position for the forehead.
    forehead_landmark = landmarks[forehead_index]

    # Convert the normalized landmark coordinates (0 to 1) to pixel coordinates.
    forehead_x = int(forehead_landmark.x * w)
    forehead_y = int(forehead_landmark.y * h)

    # Get the dimensions of the rabbit ears image.
    ear_width, ear_height = rabbit_ear_img.shape[1], rabbit_ear_img.shape[0]
    # Calculate the top-left (x1, y1) and bottom-right (x2, y2) coordinates for the rabbit ears overlay based on the forehead landmark position.
    # 💡 x1 is calculated to center the image horizontally on the forehead.
    x1, y2 = forehead_x - ear_width // 2, forehead_y
    x2, y1 = x1 + ear_width, y2 - ear_height

    ### Title: Adjust Image Boundaries
    # Ensure that the calculated coordinates do not exceed the frame boundaries.
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Calculate the width and height of the cropped image section.
    cropped_width = x2 - x1
    cropped_height = y2 - y1

    ### Title: Alpha compositing
    if cropped_width > 0 and cropped_height > 0:
        # 💡 Crop the rabbit ears image to fit within the calculated dimensions.
        rabbit_ear_cropped = rabbit_ear_img[-cropped_height:, :cropped_width]

        # Normalize the alpha channel (transparency) of the cropped image.
        # alpha_s represents source alpha (rabbit ears image).
        alpha_s = rabbit_ear_cropped[:, :, 3] / 255.0
        # alpha_l represents layer alpha (existing frame).
        alpha_l = 1.0 - alpha_s

        ## Overlay the rabbit ears image onto the frame, blending the two images together.
        # This loop applies the alpha blending to each color channel (R, G, B).
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha_s * rabbit_ear_cropped[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c]
            )


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    mode = 0  # 0: No Filter, 1: Face Detection, 2: Face Mesh, 3: Face Mesh with Rabbit Ears

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Q 키를 누르면 No Filter -> Face Detection -> Face Mesh -> Face Mesh with Rabbit Ears 전환
            if cv2.waitKey(5) & 0xFF == ord("q"):
                mode = (mode + 1) % 4
                print(
                    f"{['No Filter', 'Face Detection', 'Face Mesh', 'Face Mesh with Rabbit Ears'][mode]} Enabled"
                )

            ### 💡 Blaze Face model_selection: 0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters. See details in
            if mode == 1:
                # Face Detection 모드
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.detections:
                    annotated_image = visualize(image, results.detections)
                    cv2.imshow("Face Detection", annotated_image)
                else:
                    cv2.imshow("Face Detection", image)

            elif mode == 2:
                # Face Mesh 모드
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh_detector.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ### index10 is forehead
                        annotated_image = draw_landmarks_with_x(
                            image, face_landmarks, landmark_indices=list(range(468))
                        )
                        # annotated_image = draw_landmarks_with_x(
                        #     image, face_landmarks, landmark_indices=[10]
                        # )
                    cv2.imshow("Face Detection", annotated_image)
                else:
                    cv2.imshow("Face Detection", image)

            elif mode == 3:
                # Face Mesh with Rabbit Ears 모드
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh_detector.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        overlay_rabbit_ears(image, face_landmarks.landmark)
                    cv2.imshow("Face Detection", image)
                else:
                    cv2.imshow("Face Detection", image)

            else:
                # No Filter 모드 (원본 이미지 출력)
                cv2.imshow("Face Detection", frame)

            # ESC 키로 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
