# %%
from typing import Any

import cv2  # Import OpenCV library for image and video processing
from IPython.core.interactiveshell import (
    InteractiveShell,  # Import for IPython interactive environment settings
)
from mediapipe.python.solutions import (
    face_mesh,  # Import MediaPipe's face mesh solution for detecting facial landmarks
)
from numpy import (  # Import type definitions from NumPy
    dtype,
    floating,
    integer,
    ndarray,
)

# Set IPython to display the output of all expressions in a cell
InteractiveShell.ast_node_interactivity = "all"


def load_rabbit_ear_image(
    image_path,
) -> cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]:
    """
    Loads the rabbit ear image from the given path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        cv2.Mat | np.ndarray: The loaded image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    rabbit_ear_img = cv2.imread(
        image_path, cv2.IMREAD_UNCHANGED
    )  # Load the image, preserving the alpha channel
    if rabbit_ear_img is None:
        raise FileNotFoundError(
            f"Unable to load the image from {image_path}."
        )  # Raise an error if the image cannot be loaded
    return rabbit_ear_img  # Return the loaded image


def initialize_face_mesh(
    max_num_faces=1, min_detection_confidence=0.3, min_tracking_confidence=0.3
) -> face_mesh.FaceMesh:
    """
    Initializes the MediaPipe Face Mesh model.

    Args:
        max_num_faces (int): The maximum number of faces to process.
        min_detection_confidence (float): Minimum confidence value for the face detection to be considered reliable.
            - Used when initially detecting faces. Higher values mean the model is more confident before detecting a face.
        min_tracking_confidence (float): Minimum confidence value for the face tracking to be considered reliable.
            - Used for maintaining tracked facial landmarks. Higher values mean the model is more confident in tracking faces.

    Returns:
        face_mesh.FaceMesh: The initialized Face Mesh model.
    """
    return face_mesh.FaceMesh(
        max_num_faces=max_num_faces,  # Set the maximum number of faces to detect
        refine_landmarks=True,  # Enable refined landmark detection
        min_detection_confidence=min_detection_confidence,  # Set the minimum detection confidence threshold
        min_tracking_confidence=min_tracking_confidence,  # Set the minimum tracking confidence threshold
    )


def get_forehead_position(landmarks, frame_shape) -> tuple[int, int]:
    """
    Calculates the coordinates of the center of the forehead from facial landmarks.

    Args:
        landmarks: The facial landmarks data.
        frame_shape: The shape of the video frame (height, width).

    Returns:
        tuple[int, int]: The (x, y) coordinates of the center of the forehead.
    """
    forehead_index = (
        10  # The index for the landmark representing the center of the forehead
    )
    forehead_landmark = landmarks.landmark[forehead_index]  # Get the forehead landmark
    forehead_x = int(
        forehead_landmark.x * frame_shape[1]
    )  # Calculate the x-coordinate by scaling with the frame width
    forehead_y = int(
        forehead_landmark.y * frame_shape[0]
    )  # Calculate the y-coordinate by scaling with the frame height
    return forehead_x, forehead_y  # Return the forehead coordinates


def adjust_forehead_position(
    forehead_x, forehead_y, prev_forehead_x, prev_forehead_y
) -> tuple[int, int]:
    """
    Adjusts the forehead position to stabilize it by averaging with the previous position.

    Args:
        forehead_x (int): Current x-coordinate of the forehead.
        forehead_y (int): Current y-coordinate of the forehead.
        prev_forehead_x (int): Previous x-coordinate of the forehead.
        prev_forehead_y (int): Previous y-coordinate of the forehead.

    Returns:
        tuple[int, int]: The adjusted (x, y) coordinates of the forehead.
    """
    if prev_forehead_x is None or prev_forehead_y is None:
        return (
            forehead_x,
            forehead_y,
        )  # If no previous coordinates, return the current coordinates
    forehead_x = (
        forehead_x + prev_forehead_x
    ) // 2  # Average the current and previous x-coordinates
    forehead_y = (
        forehead_y + prev_forehead_y
    ) // 2  # Average the current and previous y-coordinates
    return forehead_x, forehead_y  # Return the adjusted coordinates


def overlay_rabbit_ears(frame, rabbit_ear_img, forehead_x, forehead_y) -> None:
    """
    Overlays the rabbit ears image onto the frame at the forehead position.

    Args:
        frame (cv2.Mat): The current video frame.
        rabbit_ear_img (cv2.Mat): The rabbit ears image to overlay.
        forehead_x (int): The x-coordinate of the forehead.
        forehead_y (int): The y-coordinate of the forehead.
    """
    ear_width, ear_height = (
        rabbit_ear_img.shape[1],
        rabbit_ear_img.shape[0],
    )  # Get the width and height of the rabbit ears image
    x1, y2 = (
        forehead_x - ear_width // 2,
        forehead_y,
    )  # Calculate the top-left corner of the overlay
    x2, y1 = (
        x1 + ear_width,
        y2 - ear_height,
    )  # Calculate the bottom-right corner of the overlay

    # Ensure the overlay is within the frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    cropped_width = x2 - x1  # Calculate the width of the cropped overlay
    cropped_height = y2 - y1  # Calculate the height of the cropped overlay

    if (
        cropped_width > 0
        and cropped_height > 0
        and cropped_width <= ear_width
        and cropped_height <= ear_height
    ):
        # Crop the rabbit ears image to fit the overlay area
        rabbit_ear_cropped = rabbit_ear_img[-cropped_height:, :cropped_width]
        alpha_s = (
            rabbit_ear_cropped[:, :, 3] / 255.0
        )  # Extract the alpha channel and normalize
        alpha_l = 1.0 - alpha_s  # Calculate the inverse alpha for the background

        # Overlay the rabbit ears image onto the frame using alpha blending
        for c in range(3):  # Loop over the color channels
            frame[y1:y2, x1:x2, c] = (
                alpha_s * rabbit_ear_cropped[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c]
            )
    else:
        # Skip overlay if the cropped size is invalid
        print(
            f"Skipping overlay: invalid cropped size {cropped_width}x{cropped_height}"
        )


def main_loop() -> None:
    """
    The main loop for processing the video feed and overlaying the rabbit ears.
    """
    rabbit_ear_img = load_rabbit_ear_image(
        "rabbit_ear2.png"
    )  # Load the rabbit ears image
    face_mesh_ = initialize_face_mesh()  # Initialize the Face Mesh model
    cap = cv2.VideoCapture(0)  # Open the default camera

    prev_forehead_x, prev_forehead_y = (
        None,
        None,
    )  # Initialize previous forehead position

    while cap.isOpened():
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:
            break  # Exit the loop if the frame cannot be captured

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # Convert the frame from BGR to RGB
        results = face_mesh_.process(
            rgb_frame
        )  # Process the frame to detect facial landmarks

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                forehead_x, forehead_y = get_forehead_position(
                    landmarks, frame.shape
                )  # Get the forehead position
                forehead_x, forehead_y = adjust_forehead_position(
                    forehead_x, forehead_y, prev_forehead_x, prev_forehead_y
                )  # Adjust the forehead position

                prev_forehead_x, prev_forehead_y = (
                    forehead_x,
                    forehead_y,
                )  # Update the previous forehead position

                overlay_rabbit_ears(
                    frame, rabbit_ear_img, forehead_x, forehead_y
                )  # Overlay the rabbit ears

        cv2.imshow("MediaPipe Face Mesh with Rabbit Ears", frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Exit the loop if 'q' is pressed

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main_loop()  # Start the main loop if the script is run directly
