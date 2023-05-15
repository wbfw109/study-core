import socketio
from mysite.server.celery_worker import celery
from mysite.server.websocket import create_socketio


app, socketio_server = create_socketio()
socketio_client = socketio.Client()


@celery.task
def add(x, y):
    print("sum")
    return x + y


@celery.task
def stream_video(event_name: str, base64str: str, namespace: str = "") -> str:
    error_message: str = ""
    socketio_server.emit(event_name, base64str, namespace=namespace, broadcast=True)

    return error_message
