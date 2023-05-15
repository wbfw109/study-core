# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


import cv2

if __name__ == "__main__":
    print("Hello")

"""
# cascade classifier == 비올라-존스 얼굴 검출 알고리즘
하르 필터(Haar-like filter) 집합
"""
## Note that Do not connect two Cameras into same Port hub. two cameras return isOpend() as 0, either but frame return value is 0.
# Search for available webcams


def show_frame(ret: bool, frame: cv2.typing.MatLike):
    if ret and frame.size > 0:
        cv2.imshow("Face Detection", frame) 
        # 키 입력 대기 (1ms) -> 'q' 또 는 'Q'를 누르면 종료
    else:
        print(index)


# %%
## Test Webcam OK
working_cam: list[cv2.VideoCapture] = []
for index in range(2):
    capture = cv2.VideoCapture(index)
    print(index, capture.isOpened())
    # if not capture.read()[0]:
    #     capture.release()
    #     continue
    # else:
    working_cam.append(capture)

# Check if the webcam is opened correctly
for index in range(2):
    if not working_cam[index].isOpened():
        raise OSError("Cannot open webcam")

# process
is_exit = False
while True:
    for index in range(2):
        ret, frame = working_cam[index].read()

        if not ret:
            print(f"Camera {index} failed to capture frame.")
            working_cam[index].release()
            working_cam[index] = cv2.VideoCapture(index)
            ret, frame = working_cam[index].read()

        if ret and frame.size > 0:
            cv2.imshow(f"Face Detection {index}", frame)
        else:
            print(f"Camera {index} returning empty frame.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            is_exit = True
            break
    if is_exit:
        break

for index in range(2):
    working_cam[index].release()
    cv2.destroyAllWindows()


# #%%

# def face_procs(
#     face_image, face_cascade: cv2.CascadeClassifier, eye_cascade: cv2.CascadeClassifier
# ):
#     gray_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

#     # 얼굴 검출
#     faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

#     # 얼굴 주위에 사각형 그리기
#     for x, y, w, h in faces:
#         cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # 얼굴 영역 추출 (ROI: Region Of Interest)
#         roi_gray = gray_img[
#             y : y + h, x : x + w
#         ]  # Region of Interest; https://numpy.org/doc/stable/user/basics.indexing.html
#         roi_color = face_image[
#             y : y + h, x : x + w
#         ]  # Region of Interest; https://numpy.org/doc/stable/user/basics.indexing.html
#         # 눈 검출
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for ex, ey, ew, eh in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


# # face_cascade = cv2.CascadeClassifier('./Images/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# eye_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
# )
# cap = cv2.VideoCapture(0)

# print(f"face_cascade.empty:{ face_cascade.empty()}")  # 파일이 비어있는지 확인
# mode_image = False
# mode_video = True


# start_time = time.time()
# #%%
# if mode_video:
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             face_procs(frame, face_cascade, eye_cascade)

#         cv2.imshow("Face Detection", frame)
#         # 키 입력 대기 (1ms) -> 'q' 또는 'Q'를 누르면 종료
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q") or key == ord("Q"):
#             break
#     capture.release()
#     cv2.destroyAllWindows()


# #%%
# if 1:

#     img_path = "../../cpp_study/resource/images/Audrey.jpg"
#     # img_path = "../../cpp_study/resource/images/Audrey.jpg"
#     img = cv2.imread(img_path)
#     """
#     rows = img.shape[0]
#     cols = img.shape[1]
#     channels = img.shape[2]
#     """
#     rows, cols, channels = img.shape
#     factor = cols / rows

#     n_cols = 400
#     n_rows = int(n_cols * factor)
#     img_resized = cv2.resize(img, (n_rows, n_cols))

#     face_image = img_resized
#     # face_procs(face_image, face_cascade, eye_cascade)
#     cv2.imshow("show_img", face_image)
#     cv2.waitKey(0)


# finish_time = time.time()
# elapsed_time_ms = (finish_time - start_time) * 1000  # ms
# print(f"Elapsed time : {elapsed_time_ms:.3f} ms")


# cv2.destroyAllWindows()


# # %%
# # TODO: dict.. key.. hash table.. search..
# hi = "ssss"

# hello = 6000
# fmt = f"{hi:10s} {hello:05d} {hello:d} world"
# fmt


# ## "\n".join( ( expression ) )
# def print_multiplication_table(
#     start_multiplier: int = 2,
#     end_multiplier: int = 9,
#     start_multiplicand: int = 2,
#     end_multiplicand: int = 9,
# ) -> None:
#     lines: str = "\n".join(
#         "".join(
#             f"{j:2d} *{i:2d} = {i*j:3d}\t"
#             for j in range(start_multiplicand, end_multiplicand + 1)
#         )
#         for i in range(start_multiplier, end_multiplier + 1)
#     )
#     print(lines)


# print_multiplication_table()
# # %%


# class MyList:
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, key):
#         return (key[0], key[1])


# my_list = MyList([10, 20, 30, 40, 50])

# # 리스트 인덱싱 (팬시 인덱싱)
# print(my_list[[0, 2, 4]])  # 출력: [10, 30, 50]

# # 튜플 인덱싱
# print(my_list[(1, 3)])  # 튜플도 리스트처럼 사용 가능, 출력: [20, 40]

# # # %%

# # %%


"""

for pyenv
sudo apt-get install -y \
    build-essential libbz2-dev libreadline-dev libsqlite3-dev \
    libssl-dev zlib1g-dev libncurses-dev libgdbm-dev libc6-dev liblzma-dev \
    tk-dev libffi-dev libnss3-dev libdb-dev libexpat1-dev libgmp-dev \
    libmpc-dev libmpfr-dev
"""