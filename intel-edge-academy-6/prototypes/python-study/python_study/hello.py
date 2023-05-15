if __name__ == "__main__":
    print("hello")

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


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 이미지 전처리
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(frame_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # 외곽선만 그리기
#             draw_connections(
#                 frame, face_landmarks.landmark, mp_face_mesh.FACEMESH_CONTOURS
#             )

#     # 화면에 비디오 표시
#     cv2.imshow("Face Mesh", frame)

#     # 종료 조건 (ESC 키를 누르면 종료)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # 웹캠 해제 및 모든 창 닫기
# cap.release()
# cv2.destroyAllWindows()
