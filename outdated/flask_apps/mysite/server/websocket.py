""" flask Server of [python Client, flask Server, browser Client]
python backend/server/websocket.py

from flask_socketio import SocketIO
    .emit()
        args – A dictionary with the JSON data to send as payload.

"""
from gevent import monkey

monkey.patch_all()
from mysite.config import CONFIG_CLASS
from mysite import create_app
import multiprocessing
from flask_socketio import SocketIO

client_count = 0
message_sum_count = 0


def create_socketio() -> tuple:
    app = create_app()

    socketio_server = SocketIO(
        app,
        async_mode=CONFIG_CLASS.ASYNC_MODE,
        # message_queue=CONFIG_CLASS.SOCKETIO_MESSAGE_QUEUE,
        # logger=CONFIG_CLASS.DEBUG,
        # engineio_logger= CONFIG_CLASS.DEBUG,
    )

    @socketio_server.on("connect", namespace="/origin")
    def connect_from_origin():
        print("[socketio_server from origin] origin Client connection established")
        socketio_server.emit(
            "request_frame_from_origin", {"data": "Connected"}, namespace="/origin"
        )

    @socketio_server.on("disconnect", namespace="/origin")
    def disconnect_from_origin():
        print("[socketio_server from origin]: origin Client disconnected")

    @socketio_server.on("response_frame_from_origin", namespace="/origin")
    def response_frame_from_origin(jpeg_base64_as_bytes_text: bytes):
        if jpeg_base64_as_bytes_text:
            # * test received data Start
            # jpg_original = base64.b64decode(jpeg_base64_as_text)
            # print("[socketio_server from origin] get data")
            # jpeg_image = cv2.imdecode(numpy.frombuffer(base64.b64decode(jpeg_base64_as_text), dtype=numpy.uint8), flags=cv2.IMREAD_COLOR)
            # cv2.imshow('test', jpeg_image)
            # key = cv2.waitKey(1)
            # * test received data End

            global client_count
            global message_sum_count
            message_sum_count += 1
            print(message_sum_count)

            if client_count > 0:
                # if it operates in server
                socketio_server.emit(
                    "get_frame",
                    {
                        "image": "data:image/jpeg;base64,{base64str}".format(
                            base64str=jpeg_base64_as_bytes_text.decode("utf-8")
                        )
                    },
                    namespace="/end",
                )

    @socketio_server.on("connect", namespace="/end")
    def connect_from_end():
        global client_count
        client_count += 1
        print("[socketio_server from end] end Client connection established")

    @socketio_server.on("disconnect", namespace="/end")
    def disconnect_from_end():
        global client_count
        client_count -= 1
        print("[socketio_server from end]: end Client disconnected")

    return (app, socketio_server)


if __name__ == "__main__":
    # * Process Start
    app, socketio_server = create_socketio()

    # * thread run
    # socketio_server.run(app, debug=CONFIG_CLASS.DEBUG, host=CONFIG_CLASS.SERVER_HOST)

    # * process run
    processes = [
        multiprocessing.Process(
            target=socketio_server.run,
            args=(app,),
            kwargs=dict(debug=CONFIG_CLASS.DEBUG, host=CONFIG_CLASS.SERVER_HOST),
        ),
    ]
    [p.start() for p in processes]

    # * Process End
