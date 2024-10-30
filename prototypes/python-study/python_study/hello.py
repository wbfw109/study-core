import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Test Window", frame)

            # 종료 조건 (ESC 키를 누르면 종료)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # print("hello")
    # # Tkinter 윈도우 생성
    # root = tk.Tk()
    # root.title("SSH X11 Forwarding Test")

    # # 라벨 추가
    # label = tk.Label(root, text="Tkinter X11 Forwarding Test")
    # label.pack(padx=20, pady=20)

    # # 윈도우 실행
    # root.mainloop()

# # %%
# import cv2
# from mediapipe.framework.formats import landmark_pb2
# from mediapipe.python.solutions import drawing_utils, face_mesh

# # 토끼귀 스티커 이미지 로드 (투명 배경 PNG 파일)
# sticker = cv2.imread("../resources/bunny_ears.webp", cv2.IMREAD_UNCHANGED)  # RGBA

# # MediaPipe 초기화
# mp_face_mesh = face_mesh
# mp_drawing = drawing_utils

# face_mesh = mp_face_mesh.FaceMesh()

# # green
# drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)


# def draw_connections(
#     frame: cv2.Mat,
#     landmarks: landmark_pb2.NormalizedLandmarkList,
#     connections: list[Tuple[int, int]],
# ) -> None:
#     for connection in connections:
#         start_idx, end_idx = connection
#         start_landmark = landmarks[start_idx]
#         end_landmark = landmarks[end_idx]
#         start_point = (
#             int(start_landmark.x * frame.shape[1]),
#             int(start_landmark.y * frame.shape[0]),
#         )
#         end_point = (
#             int(end_landmark.x * frame.shape[1]),
#             int(end_landmark.y * frame.shape[0]),
#         )
#         cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
