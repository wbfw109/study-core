import cv2

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set the video format to MJPG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Video stream opened successfully.")
    # Loop to continuously capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow("Webcam Video", frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
