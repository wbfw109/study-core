import socket
from typing import Final


class DevConfig:
    SOCK_BUFFER_SIZE: Final[int] = 4096
    SOCK_TYPE = socket.SOCK_STREAM
