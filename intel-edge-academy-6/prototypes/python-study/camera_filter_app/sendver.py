import sys

import cv2
from mediapipe.python.solutions import face_mesh
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera_filter_app.config.paths import AssetPaths
from camera_filter_app.filters import (
    add_blush,
    apply_background_change,
    overlay_rabbit_ears,
    overlay_sunglasses,
)

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(),  # 모델 경로 없이 기본 모델 사용
    num_faces=10,
    output_face_blendshapes=False,  # 블렌드셰이프 출력 비활성화
    running_mode=vision.RunningMode.LIVE_STREAM,
)

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

face_landmarker2 = FaceLandmarker.create_from_options(options)


class CameraApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Camera App with Filters")

        # MediaPipe 초기화
        self.mp_face_mesh = face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=10)

        # GUI setup
        self.initUI()

        # Video capture object
        self.cap = cv2.VideoCapture(0)
        self.zoom_factor: float = 1.0
        self.drag_start: QPoint | None = None
        self.start_stop_camera: bool = False

        # 필터 토글 상태
        self.filter_states: dict[str, bool] = {
            "Blush": False,
            "Sunglasses": False,
            "RabbitEars": False,  # 토끼 귀 필터 추가
            "Background": False,  # 배경 필터 상태 추가
        }

        # 배경 이미지 설정
        self.background_images: list[str] = [
            f"{AssetPaths.BACKGROUND_SPACE_PATH}",
            f"{AssetPaths.BACKGROUND_OCEAN_PATH}",
            f"{AssetPaths.BACKGROUND_PHOTO_ZONE_PATH}",
        ]
        self.current_background: str | None = None

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self) -> None:
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.resize(800, 600)

        # Set up QGraphicsView to display the camera stream
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Enable drag mode

        # Zoom In/Out buttons
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        # Camera Start/Stop buttons
        self.start_btn = QPushButton("Camera Start")
        self.start_btn.clicked.connect(self.toggle_start_camera)

        # Capture Image button
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)

        # Filter buttons
        self.toggle_blush_btn = QPushButton("Toggle Blush")
        self.toggle_blush_btn.setCheckable(True)
        self.toggle_blush_btn.toggled.connect(lambda: self.toggle_filter("Blush"))

        self.toggle_sunglasses_btn = QPushButton("Toggle Sunglasses")
        self.toggle_sunglasses_btn.setCheckable(True)
        self.toggle_sunglasses_btn.toggled.connect(
            lambda: self.toggle_filter("Sunglasses")
        )

        # Toggle Rabbit Ears button
        self.toggle_rabbit_ears_btn = QPushButton("Toggle Rabbit Ears")
        self.toggle_rabbit_ears_btn.setCheckable(True)
        self.toggle_rabbit_ears_btn.toggled.connect(
            lambda: self.toggle_filter("RabbitEars")
        )

        # Toggle Background button
        self.toggle_background_btn = QPushButton("Space")
        self.toggle_background_btn.setCheckable(True)
        self.toggle_background_btn.toggled.connect(
            lambda checked: self.toggle_background(checked, index=0)
        )

        self.toggle_background2_btn = QPushButton("Ocean")
        self.toggle_background2_btn.setCheckable(True)
        self.toggle_background2_btn.toggled.connect(
            lambda checked: self.toggle_background(checked, index=1)
        )

        self.toggle_background3_btn = QPushButton("Photo zone")
        self.toggle_background3_btn.setCheckable(True)
        self.toggle_background3_btn.toggled.connect(
            lambda checked: self.toggle_background(checked, index=2)
        )

        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.toggle_background_btn)
        button_layout2.addWidget(self.toggle_background2_btn)
        button_layout2.addWidget(self.toggle_background3_btn)

        # Layout setup
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.toggle_blush_btn)
        button_layout.addWidget(self.toggle_sunglasses_btn)
        button_layout.addWidget(self.toggle_rabbit_ears_btn)

        zoom = QHBoxLayout()
        zoom.addWidget(self.zoom_in_btn)
        zoom.addWidget(self.zoom_out_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.capture_btn)
        layout.addLayout(zoom)
        layout.addLayout(button_layout)
        layout.addLayout(button_layout2)

        self.central_widget.setLayout(layout)

    def toggle_start_camera(self) -> None:
        self.start_stop_camera = not self.start_stop_camera
        if self.start_stop_camera:
            self.timer.start(30)
            self.start_btn.setText("Camera Stop")
        else:
            self.timer.stop()
            self.start_btn.setText("Camera Start")

    def stop_camera(self) -> None:
        self.timer.stop()

    def zoom_in(self) -> None:
        self.zoom_factor *= 1.1
        self.view.scale(1.1, 1.1)

    def zoom_out(self) -> None:
        self.zoom_factor *= 0.9
        self.view.scale(0.9, 0.9)

    def toggle_filter(self, filter_name: str) -> None:
        self.filter_states[filter_name] = not self.filter_states[filter_name]

    def toggle_background(self, checked: bool, index: int) -> None:
        if checked:
            self.current_background = self.background_images[index]
            self.filter_states["Background"] = True
        else:
            self.current_background = None
            self.filter_states["Background"] = False

    def update_frame(self) -> None:
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            if any(self.filter_states.values()):  # 필터가 하나라도 켜져 있으면
                frame = self.apply_face_filter(frame)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            t_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(t_image)

            self.scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)

            self.current_frame = frame
        else:
            self.scene.clear()

    def apply_face_filter(self, frame: cv2.Mat) -> cv2.Mat:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.filter_states["Blush"]:
                    add_blush(frame, face_landmarks.landmark)
                if self.filter_states["Sunglasses"]:
                    overlay_sunglasses(frame, face_landmarks.landmark)
                if self.filter_states["RabbitEars"]:
                    overlay_rabbit_ears(frame, face_landmarks.landmark)
                if self.filter_states["Background"] and self.current_background:
                    bg_image = cv2.imread(self.current_background)
                    frame = apply_background_change(
                        frame, face_landmarks.landmark, bg_image
                    )

        return frame

    def capture_image(self) -> None:
        ## 느
        if hasattr(self, "current_frame"):
            cv2.imwrite("./Images/capture_image.png", self.current_frame)
            print("Image captured and saved as 'capture_image.png'")
            captured_img = cv2.imread("./Images/capture_image.png", cv2.IMREAD_COLOR)
            cv2.imshow("Captured Image", captured_img)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_start:
            delta = event.pos() - self.drag_start
            self.view.horizontalScrollBar().setValue(
                self.view.horizontalScrollBar().value() - delta.x()
            )
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().value() - delta.y()
            )
            self.drag_start = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.drag_start = None
        super().mouseReleaseEvent(event)


if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    window = CameraApp()
    window.show()
    sys.exit(app.exec())
