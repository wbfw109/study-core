from pathlib import Path, PurePath

# * GLOBAL config Start
# see the last part of this file for global configs that to be changed based on global settings.
# APP_DEVELOPMENT_MODE = {"d", "t", "p"}. this global variable only used to initialize CONFIG_CLASS
APP_DEVELOPMENT_MODE = "d"
# * GLOBAL config End


class Config(object):
    PROJECT_ABSOLUTE_ROOT_DIRECTORY: Path = Path(
        "/home/wbfw109/study-core/study-flask-with_tensorflow_v1"
    )

    DEBUG = False
    TESTING = False
    # SERVER_HOST = {"127.0.0.1", "0.0.0.0"}
    SERVER_HOST = "127.0.0.1"
    SOCKETIO_MESSAGE_QUEUE = "redis://:{password}@{url}".format(
        password="root", url="127.0.0.1:6379"
    )
    SECRET_KEY = b"a\x16s$c{'@\r\xf1[~x\xd85\xf2"
    SESSION_COOKIE_SECURE = True

    # ASYNC_MODE = {"threading", "gevent"}
    ASYNC_MODE = "threading"

    # * resource setting Start
    GOOGLE_DRIVE_APP_PATH = Path("/mnt/c/Users/wbfw109/MyDrive")
    GOOGLE_DRIVE_APP_SYMBOLIC_LINK_Path: Path = Path.home() / "my_google_drive_app"
    DEFAULT_FLASK_PATH = PurePath("starter")
    DEFAULT_LOCAL_FILE_PATH = Path.home() / "wbfw109_flask"
    IMAGE_PROCESSING_ROOT_PATH = DEFAULT_LOCAL_FILE_PATH / "image_processing"
    MACHINE_LEARNING_ROOT_PATH = DEFAULT_LOCAL_FILE_PATH / "ml"
    ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".wmv", ".flv"}
    IMAGE_PROCESSING_SOURCE_PATH = IMAGE_PROCESSING_ROOT_PATH / "source"
    IMAGE_PROCESSING_RESULT_PATH = IMAGE_PROCESSING_ROOT_PATH / "result"
    STATIC_FOLDER = DEFAULT_FLASK_PATH / "static"
    UPLOAD_FOLDER = STATIC_FOLDER / "uploads"
    RESOURCE_FOLDER = DEFAULT_FLASK_PATH / "resource"
    MAX_CONTENT_LENGTH = 16 * 1000 * 1000

    # ** utility setting Start
    HOLIDAYS_IN_SOUTH_KOREA_PATH: Path = (
        RESOURCE_FOLDER / "holidays_in_south_korea.json"
    )
    # ** utility setting End

    # * resource setting End

    # celery setting
    CELERY_BROKER_URL = "redis://:{password}@{url}/{database}".format(
        password="root", url="localhost:6379", database=0
    )
    CELERY_RESULT_BACKEND = "redis://:{password}@{url}/{database}".format(
        password="root", url="localhost:6379", database=0
    )

    @classmethod
    def ensure_exists_folder(cls) -> None:
        Path(cls.DEFAULT_LOCAL_FILE_PATH).mkdir(exist_ok=True)
        Path(cls.MACHINE_LEARNING_ROOT_PATH).mkdir(exist_ok=True)
        Path(cls.IMAGE_PROCESSING_ROOT_PATH).mkdir(exist_ok=True)
        Path(cls.IMAGE_PROCESSING_SOURCE_PATH).mkdir(exist_ok=True)
        Path(cls.IMAGE_PROCESSING_RESULT_PATH).mkdir(exist_ok=True)
        Path(cls.STATIC_FOLDER).mkdir(exist_ok=True)
        Path(cls.UPLOAD_FOLDER).mkdir(exist_ok=True)
        Path(cls.RESOURCE_FOLDER).mkdir(exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    TEMPLATES_AUTO_RELOAD = True
    ASYNC_MODE = "gevent"
    # only in tutorial
    # DATABASE=os.path.join(app.instance_path, "tutorial.sqlite")


class TestingConfig(Config):
    TESTING = True
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    ASYNC_MODE = "gevent"
    pass


if APP_DEVELOPMENT_MODE == "d":
    CONFIG_CLASS = DevelopmentConfig
elif APP_DEVELOPMENT_MODE == "t":
    CONFIG_CLASS = TestingConfig
elif APP_DEVELOPMENT_MODE == "p":
    CONFIG_CLASS = ProductionConfig
