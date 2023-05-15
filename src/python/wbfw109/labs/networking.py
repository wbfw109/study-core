# %%
from __future__ import annotations

import functools
import selectors
import socket
import threading
import time
from typing import (
    Final,
    LiteralString,
    Optional,
)

from IPython.core.interactiveshell import InteractiveShell
from wbfw109.libs.utilities.ipython import (
    VisualizationManager,
    VisualizationRoot,
)

# + allow multiple print
InteractiveShell.ast_node_interactivity = "all"

# %doctest_mode


#%%


class SockEchoCommunication(VisualizationRoot):
    """
    https://docs.python.org/3/library/socket.html
    https://docs.python.org/3/library/selectors.html#module-selectors
    https://docs.python.org/3/howto/sockets.html#socket-howto
    """

    SOCK_BUFFER_SIZE: Final[int] = 4096
    SOCK_TYPE = socket.SOCK_STREAM

    def __init__(self) -> None:
        VisualizationRoot.__init__(
            self, columns=["position", "print"], has_df_lock=True, should_highlight=True
        )

        self.MAX_TRAFFIC_COUNT: Final[int] = 2
        self.SERVER_SOCK_NAME: str = "➕ server"
        self._lock_connection_count = threading.Lock()
        self.is_server_open: bool = False
        self.selectors: selectors.EpollSelector = selectors.DefaultSelector()  # type: ignore
        self.client_map_to_traffic_count: dict[tuple[str, int], int] = {}
        self.connection_count: int = 0

    def __str__(self) -> str:
        return "(Stream type, muxing, Non-blocking)"

    def echo(self, server_sock: socket.socket) -> None:
        """run when server sock receives client socket data"""
        remote_peer_address: tuple[str, int] = server_sock.getpeername()
        data = server_sock.recv(SockEchoCommunication.SOCK_BUFFER_SIZE)
        server_sock.send(data)

        # Traffic limit by remote peer
        self.client_map_to_traffic_count[remote_peer_address] += 1
        if (
            self.client_map_to_traffic_count[remote_peer_address]
            >= self.MAX_TRAFFIC_COUNT
        ):
            self.selectors.unregister(server_sock)
            server_sock.close()
            del self.client_map_to_traffic_count[remote_peer_address]
            with self._lock_connection_count:
                self.connection_count -= 1
            self.append_line_into_df_in_wrap(
                [
                    self.SERVER_SOCK_NAME,
                    f"⚠️ [Traffic Limit] close connection {remote_peer_address}. sum count: {self.connection_count}",
                ]
            )

    def accept_client_sock(self, sock: socket.socket) -> None:
        """run when server sock in listen accepts client socket connection"""
        with self._lock_connection_count:
            self.connection_count += 1
        remote_peer_sock, remote_peer_address = sock.accept()
        remote_peer_sock.setblocking(False)
        self.client_map_to_traffic_count[remote_peer_address] = 0
        self.selectors.register(remote_peer_sock, selectors.EVENT_READ, self.echo)
        self.append_line_into_df_in_wrap(
            [
                self.SERVER_SOCK_NAME,
                f"Catch connection {remote_peer_address}. sum count: {self.connection_count}",
            ]
        )

    def run_echo_server(self) -> None:
        HOST: Optional[LiteralString] = None
        PORT: Final[int] = 50007  # Arbitrary non-privileged port
        LISTEN_BACKLOG: Final[int] = 3
        SERVER_LIFE_SECONDS: Final[float] = 1.5
        SERVER_READ_EVENT_TIMEOUT_CYCLE: Final[float] = SERVER_LIFE_SECONDS / 10
        server_sock: Optional[socket.socket] = None

        for (
            address_family,
            sock_type,
            proto,
            cname,  # type: ignore
            sock_address,
        ) in socket.getaddrinfo(  # type :ignore
            HOST,
            PORT,
            socket.AF_UNSPEC,
            SockEchoCommunication.SOCK_TYPE,
            0,
            socket.AI_PASSIVE,
        ):
            try:
                server_sock = socket.socket(address_family, sock_type, proto)
            except OSError:  # type: ignore
                server_sock = None
                continue
            try:
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind(sock_address)
                server_sock.listen(LISTEN_BACKLOG)
            except OSError:  # type: ignore
                server_sock.close()
                server_sock = None
                continue
            break
        if server_sock is None:
            self.append_line_into_df_in_wrap(
                [self.SERVER_SOCK_NAME, "🚨 [Error] Could not open socket"]
            )
            return
        self.is_server_open = True
        self.append_line_into_df_in_wrap(
            [
                self.SERVER_SOCK_NAME,
                "Waiting for an incoming connection...",
            ]
        )

        # Set blocking to false so that program can send and receive messages at the same time
        server_sock.setblocking(False)
        self.selectors.register(
            server_sock, selectors.EVENT_READ, self.accept_client_sock
        )
        elapsed_time_in_no_data: float = 0.0
        while True:
            start_time = time.time()
            events = self.selectors.select(SERVER_READ_EVENT_TIMEOUT_CYCLE)
            for key, mask in events:  # type: ignore
                elapsed_time_in_no_data = 0.0
                callback = key.data
                callback(key.fileobj)
            elapsed_time_in_no_data += time.time() - start_time
            if elapsed_time_in_no_data >= SERVER_LIFE_SECONDS:
                self.selectors.close()
                break

    def run_echo_remote_peer(self, name: str):
        MY_SOCK_NAME: str = f"➖ remote peer: {name}"
        SERVER_HOST: LiteralString = "localhost"  # The remote host
        SERVER_PORT: Final[int] = 50007  # The same port as used by the server
        sock: Optional[socket.socket] = None
        send_data_list: list[bytes] = [
            (data + name).encode("utf-8") for data in ["Hello ", ", World ", "Bye "]
        ]

        received_data_list: list[str] = []
        if not self.is_server_open:
            time.sleep(0.2)
        for address_family, sock_type, proto, cname, sock_address in socket.getaddrinfo(  # type: ignore
            SERVER_HOST, SERVER_PORT, socket.AF_UNSPEC, SockEchoCommunication.SOCK_TYPE
        ):
            try:
                sock = socket.socket(address_family, sock_type, proto)
            except OSError as msg:
                print("client:", msg)
                sock = None
                continue
            try:
                sock.connect(sock_address)
            except OSError as msg:
                print("client:", msg)
                sock.close()
                sock = None
                continue
            break
        if sock is None:
            self.append_line_into_df_in_wrap([MY_SOCK_NAME, "could not open socket"])
            return
        with sock:
            for send_data in send_data_list:
                sock.sendall(send_data)
                try:
                    data = sock.recv(SockEchoCommunication.SOCK_BUFFER_SIZE)
                except ConnectionResetError:
                    # Server requests close.
                    data = None

                # Note that Client socket must be closed or break in loop under "with" statement if no data, otherwise forever waiting.
                if not data:
                    break
                received_data_list.append(repr(data))
            self.append_line_into_df_in_wrap(
                [
                    MY_SOCK_NAME,
                    f"Send data: ❕ {' ❕ '.join(list(map(repr, send_data_list)))}",
                ]
            )
            self.append_line_into_df_in_wrap(
                [MY_SOCK_NAME, f"Received data: ❕ {' ❕ '.join(received_data_list)}"]
            )

    @classmethod
    def test_case(cls) -> None:  # type: ignore
        # 1 server and 3 remote peer
        socket_echo_communication = SockEchoCommunication()
        t_echo_server = threading.Thread(
            target=socket_echo_communication.run_echo_server
        )
        t_echo_remote_peer_list: list[threading.Thread] = [
            t_echo_server,
            *[
                threading.Thread(
                    target=functools.partial(
                        socket_echo_communication.run_echo_remote_peer, name=str(name)
                    )
                )
                for name in range(1, 4)
            ],
        ]
        for thread in t_echo_remote_peer_list:
            thread.start()
        for thread in t_echo_remote_peer_list:
            thread.join(timeout=None)
        socket_echo_communication.visualize()


if __name__ == "__main__" or VisualizationManager.central_control_state:
    if VisualizationManager.central_control_state:
        # Do not change this.
        only_class_list = []
    else:
        only_class_list = []
    VisualizationManager.call_root_classes(only_class_list=only_class_list)
