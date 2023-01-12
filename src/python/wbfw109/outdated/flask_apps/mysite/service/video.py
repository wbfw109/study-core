from mysite.config import CONFIG_CLASS
from typing import Generator, Union
from pathlib import Path
import datetime
import cv2
import numpy


class LocalImages:
    def __init__(self, path_image_dir: Path):
        images_path: list = [
            image
            for image in path_image_dir.iterdir()
            if image.suffix.lower() in CONFIG_CLASS.ALLOWED_IMAGE_EXTENSIONS
        ]

        self.images_path: list = []
        self.original_images: list = []
        for image_path in images_path:
            if (not_none_image := cv2.imread(str(image_path))) is not None:
                self.images_path.append(image_path)
                self.original_images.append(not_none_image)

        self.buffered_images: list = self.original_images
        self.mode_arguments: dict = {}

    def _resize_base_on_min_width(self, images: tuple) -> list:
        images_width: list = [numpy.size(image, 1) for image in images]
        if len(set(images_width)) == 1:
            return images
        else:
            min_width: int = min(images_width)
            return [
                cv2.resize(image, dsize=(min_width, numpy.size(image, 0)))
                for image in images
            ]

    def _resize_base_on_min_height(self, images: tuple) -> list:
        images_height: list = [numpy.size(image, 0) for image in images]
        if len(set(images_height)) == 1:
            return images
        else:
            min_height: int = min(images_height)
            return [
                cv2.resize(image, dsize=(numpy.size(image, 1), min_height))
                for image in images
            ]

    def crop(self, mode: str) -> None:
        # pre-processing
        new_images: list = []
        # input
        self.mode_arguments[mode] = list(
            map(
                int,
                input(
                    "Enter the crop size: (width, height). \n  for example, width 1 ~ 100, height 21 ~ 200 input is 0 100 21 200  :  "
                ).split(),
            )
        )

        # # for test input
        # self.mode_arguments[mode] = [1, 200, 100, 300]

        if len(self.mode_arguments[mode]) != 4:
            raise TypeError(
                "TypeError: {mode} mode takes 4 arguments. but {count} were given.".format(
                    mode=mode, count=len(self.mode_arguments[mode])
                )
            )

        # process
        for image in self.buffered_images:
            # pre-processinging
            new_image_size: list = []

            # * validates size Start
            new_image_size.append(self.mode_arguments[mode][0] % numpy.size(image, 1))
            new_image_size.append(self.mode_arguments[mode][1] % numpy.size(image, 1))
            if new_image_size[1] == 0:
                new_image_size[1] = numpy.size(image, 1)
            new_image_size.append(self.mode_arguments[mode][2] % numpy.size(image, 0))
            new_image_size.append(self.mode_arguments[mode][3] % numpy.size(image, 0))
            if new_image_size[3] == 0:
                new_image_size[3] = numpy.size(image, 0)

            if (
                new_image_size[0] == new_image_size[1]
                or new_image_size[2] == new_image_size[3]
            ):
                raise Exception(
                    "ValueError: width and height value must be greater than 0."
                )

            # * validates size End
            # image[<height start>:<height end>, <width start>:<width end>])
            new_images.append(
                image[
                    new_image_size[2] : new_image_size[3],
                    new_image_size[0] : new_image_size[1],
                ]
            )

            # post-processing
            self.buffered_images = new_images

    def resize(self, mode: str) -> None:
        # pre-processing
        new_images: list = []

        resize_mode: str = input(
            "Enter the resize mode: desired size (width, height) or scaling factor value (fx, fy).  'd' or 'f' :  "
        )
        if resize_mode == "d":
            self.mode_arguments[mode] = list(
                map(
                    int,
                    input(
                        "Enter values in form like : (width, height). these value must be >=1 and < 4000 in this function. \n  for example, width 500, height 2000 input is 500 2000  :  "
                    ).split(),
                )
            )
            for value in self.mode_arguments[mode]:
                if value < 1 or value > 4000:
                    raise ValueError("Value must be >= 1 and <= 4000")

        elif resize_mode == "f":
            self.mode_arguments[mode] = list(
                map(
                    float,
                    input(
                        "Enter values in form like : (X-axix scale factor, Y-axis scale factor). these value must be > 0.0 and <= 4.0 in this function. \n  for example, 0.5, height 0.8 input is 0.5 0.8  :  "
                    ).split(),
                )
            )
            for value in self.mode_arguments[mode]:
                if value <= 0 or value > 4.0:
                    raise ValueError("Value must be >= 0 and < 4000")
        else:
            raise Exception(
                "{input_string} is impossible mode. it must be 'd' or 'f'.".format(
                    input_string=resize_mode
                )
            )

        if len(self.mode_arguments[mode]) != 2:
            raise TypeError(
                "TypeError: crop {mode} takes 2 arguments. but {count} were given.".format(
                    mode=mode, count=len(self.mode_arguments[mode])
                )
            )

        # process
        for image in self.buffered_images:
            if resize_mode == "d":
                new_images.append(
                    cv2.resize(
                        image,
                        dsize=(
                            self.mode_arguments[mode][0],
                            self.mode_arguments[mode][1],
                        ),
                    )
                )
            elif resize_mode == "f":
                new_images.append(
                    cv2.resize(
                        image,
                        dsize=None,
                        fx=self.mode_arguments[mode][0],
                        fy=self.mode_arguments[mode][1],
                    )
                )

        # post-processing
        self.buffered_images: list = new_images

    def convert_to_grayscale(self, mode: str) -> None:
        # pre-processing
        new_images: list = []

        # process
        for image in self.buffered_images:
            new_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # post-processing
        self.buffered_images: list = new_images

    def rotate(self, mode: str) -> None:
        # pre-processing
        new_images: list = []
        angle: float = -float(input("Enter the angle :  "))

        # process
        for image in self.buffered_images:
            # * https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
            # Affine transformation: 점, 직선, 평면을 보존하는 선형 매핑 방법
            image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            new_images.append(
                cv2.warpAffine(
                    image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
                )
            )

        # post-processing
        self.buffered_images: list = new_images

    def merge(self, mode: str) -> None:
        # pre-processing
        new_images: list = []

        # process
        for image in self.buffered_images:
            blue, green, red = cv2.split(image)
            # # for test. since the separated channels are a single channel, they are expressed in black and white color.
            # cv2.imshow("blue", blue)
            # cv2.imshow("red", red)
            # cv2.imshow("green", green)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # src_data: array of pointers to source arrays (cn items x len items) [ [B, B, ...], [G, G, ...], [R, R, ...] ]
            new_images.append(cv2.merge((blue, red, green)))

        # post-processing
        self.buffered_images: list = new_images

    def blend(self, mode: str) -> None:
        # linear blend operator: g(x) = (1−α)*f0(x) + α*f1(x)
        # Since we are adding src1 and src2, they both have to be of the same size (width and height) and type
        # pre-processing
        new_images: list = []

        alpha: float = float(
            input("Enter the alpha value for linear blend (0 <= alpha <= 1) :  ")
        )
        if alpha > 1 or alpha < 0:
            raise Exception(
                "Error: value for linear blend must be 0 or more and 1 or less"
            )
        else:
            beta: float = 1 - alpha

        # process
        for index in range(len(self.buffered_images)):
            next_index = (index + 1) % len(self.buffered_images)
            resized_images: tuple = self._resize_base_on_min_width(
                self._resize_base_on_min_height(
                    (self.buffered_images[index], self.buffered_images[next_index])
                )
            )
            new_images.append(
                cv2.addWeighted(
                    resized_images[0],
                    alpha,
                    resized_images[1],
                    beta,
                    0.0,
                )
            )

        # post-processing
        self.buffered_images: list = new_images

    def hstack(self, mode: str) -> None:
        # all of the matrices must have the same number of rows and the same depth.
        # pre-processing
        new_images: list = []

        # process
        # stack operations: numpy.tile
        for index in range(len(self.buffered_images)):
            next_index = (index + 1) % len(self.buffered_images)

            # resize min or max height (min in this function)

            new_images.append(
                cv2.hconcat(
                    self._resize_base_on_min_height(
                        (self.buffered_images[index], self.buffered_images[next_index])
                    )
                )
            )

        # post-processing
        self.buffered_images: list = new_images

    def vstack(self, mode: str) -> None:
        # all of the matrices must have the same number of rows and the same depth.
        # pre-processing
        new_images: list = []

        # process
        # stack operations: numpy.tile
        for index in range(len(self.buffered_images)):
            next_index = (index + 1) % len(self.buffered_images)

            # resize min or max width (min in this function)
            new_images.append(
                cv2.vconcat(
                    self._resize_base_on_min_width(
                        (self.buffered_images[index], self.buffered_images[next_index])
                    )
                )
            )

        # post-processing
        self.buffered_images: list = new_images

    def operate(self, modes: list) -> None:
        if self.original_images:
            for mode in modes:
                if mode == "crop":
                    self.crop(mode)
                elif mode == "resize":
                    self.resize(mode)
                elif mode == "grayscale":
                    self.convert_to_grayscale(mode)
                elif mode == "rotation":
                    self.rotate(mode)
                elif mode == "merge":
                    self.merge(mode)
                elif mode == "blend":
                    self.blend(mode)
                elif mode == "hstack":
                    self.hstack(mode)
                elif mode == "vstack":
                    self.vstack(mode)
                else:
                    print("not supported mode: {mode}".format(mode))

            operations: str = "_".join(modes)
            for index, new_image in enumerate(self.buffered_images):
                image_name = "{default_filename}_{operations}_{time}{extension}".format(
                    default_filename="".join(
                        self.images_path[index].name.rsplit(
                            self.images_path[index].suffix, 1
                        )
                    ),
                    operations=operations,
                    time=datetime.now().strftime("%Y_%m%d_%H%M%m%f"),
                    extension=self.images_path[index].suffix,
                )
                cv2.imwrite(
                    str(CONFIG_CLASS.IMAGE_PROCESSING_RESULT_PATH / image_name),
                    new_image,
                )

            if args.quiet:
                pass
            elif args.verbosity >= 2:
                # print("the square of {} equals {}".format(args.square, modes))
                # os.path.splittext()
                print(
                    "sum of images: {sum_of_images}, operations: {operations}".format(
                        sum_of_images=len(self.images_path), operations=modes
                    )
                )
            else:
                # print("{}^2 == {}".format(args.square, modes))
                print(
                    "sum of images: {sum_of_images}".format(
                        sum_of_images=len(self.images_path)
                    )
                )


class LocalVideos:
    def __init__(
        self,
        directory: Path = CONFIG_CLASS.IMAGE_PROCESSING_SOURCE_PATH,
        file: Path = Path(),
        index: int = -1,
    ):
        """LocalVideos

        Args:
            - directory (Path, optional): directory that exists video files. Defaults to CONFIG_CLASS.IMAGE_PROCESSING_SOURCE_PATH = Path.home() / "image_processing" / "source"
            - file (Path, optional): video file. Defaults to Path() (not used)
            - index (int, optional): id of the video capturing device to open. Defaults to -1 (not used)
        """

        self.directory: list = []
        self.file = file
        self.index = index

        videos_path: list = [
            video
            for video in directory.iterdir()
            if video.suffix.lower() in CONFIG_CLASS.ALLOWED_VIDEO_EXTENSIONS
        ]

        for video_path in videos_path:
            try:
                video_capture = cv2.VideoCapture(str(video_path))
            except Exception as e:
                print(e)
            else:
                if video_capture.isOpened():
                    self.directory.append(str(video_path))
                else:
                    print(
                        "Skip reading corrupted file: {video_path}".format(
                            video_path=video_path
                        )
                    )
            finally:
                video_capture.release()

        if (
            file != Path()
            and file.suffix.lower() in CONFIG_CLASS.ALLOWED_VIDEO_EXTENSIONS
        ):
            try:
                video_capture = cv2.VideoCapture(str(file))
            except Exception as e:
                print(e)
            else:
                if video_capture.isOpened():
                    self.file = str(file)
                else:
                    print("Skip reading corrupted file: {file}".format(file=str(file)))
            finally:
                video_capture.release()

    @staticmethod
    def generator_of_frame(source: Union[str, int]):
        """generate frame encoded as jpeg from a video using cv2.VideoCapture()

        Args:
            source (str): avaliable type: file, device id

        Yields:
            numpy.ndarray: image frame encoded as jpeg
        """
        try:
            capture = cv2.VideoCapture(source)
            # loop over frames from the output stream
            while capture.isOpened():
                read_return_value, frame = capture.read()
                # check if the output frame is available, otherwise break the iteration of the loop
                if not read_return_value:
                    break
                # encode the frame in JPEG format
                (frame_return_value, encoded_image) = cv2.imencode(".jpeg", frame)
                # ensure the frame was successfully encoded
                if not frame_return_value:
                    continue
                yield encoded_image
        finally:
            capture.release()

    def generator_as_form_of_generic_message(self):
        """
        if self.index is -1, generate frame from self.index

        elif self.video_path is not Path(), generate frame from self.video_path,

        else (default) generate frame from self.path_video_dir

        Warning!
            * This only works well in when use it Werkzeug including socketio not with message broker.
            Otherwise, server will be stopped until the server exits.

        Yields:
            b'str: general-message
        """
        if self.index != -1:
            for frame in (
                lambda: (yield from LocalVideos.generator_of_frame(self.index))
            )():
                yield (
                    b"".join(
                        [
                            b"--frame\r\n",
                            b"Content-Type: image/jpeg\r\n\r\n",
                            bytearray(frame),
                            b"\r\n",
                        ]
                    )
                )
        elif self.file != Path():
            for frame in (
                lambda: (yield from LocalVideos.generator_of_frame(self.file))
            )():
                yield (
                    b"".join(
                        [
                            b"--frame\r\n",
                            b"Content-Type: image/jpeg\r\n\r\n",
                            bytearray(frame),
                            b"\r\n",
                        ]
                    )
                )
        else:
            for video_path in self.directory:
                for frame in (
                    lambda: (yield from LocalVideos.generator_of_frame(video_path))
                )():
                    yield (
                        b"".join(
                            [
                                b"--frame\r\n",
                                b"Content-Type: image/jpeg\r\n\r\n",
                                bytearray(frame),
                                b"\r\n",
                            ]
                        )
                    )
