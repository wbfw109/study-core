from flask import Blueprint
from mysite.service.video import LocalVideos
import asyncio
from flask import (
    stream_with_context,
    request,
    Blueprint,
    current_app,
    Response,
    render_template,
)
import time

streaming_bp = Blueprint("stream", __name__, url_prefix="/stream")


@streaming_bp.route("/")
def index():
    return render_template("stream/index.html")


@streaming_bp.route("/video_feed_my_camera")
def video_feed_my_camera():
    local_videos = LocalVideos(index=0)
    return Response(
        local_videos.generator_as_form_of_generic_message(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@streaming_bp.route("/video_feed")
def video_feed():
    """
    multipart/x-mixed-replace
        - Required parameters : boundary (defined in RFC2046)
        - Security considerations : Subresources of a multipart/x-mixed-replace resource can be of any type, including types with non-trivial security implications such as text/html.
    """
    local_videos = LocalVideos()

    return Response(
        local_videos.generator_as_form_of_generic_message(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@streaming_bp.route("/websocket/")
def websocket_index():
    return render_template("stream/websocket_index.html")


@streaming_bp.route("/test_chat/")
def test_socket():
    return render_template("test/test_socket.html")


# test ~
def stream_template(template_name, **context):
    current_app.update_template_context(context)
    template = current_app.jinja_env.get_template(template_name)
    return_value = template.stream(context)
    return_value.enable_buffering(5)
    return return_value


"""
위에꺼 확인 후 비디오 파일 재생 하는 함수, 현재 카메라 재생하는 함수부터 작성




javascript
  https://joshua1988.github.io/web-development/javascript/js-async-await/
  async와 await는 자바스크립트의 비동기 처리 패턴 중 가장 최근에 나온 문법입니다. 기존의 비동기 처리 방식인 콜백 함수와 프로미스의 단점을 보완하고 개발자가 읽기 좋은 코드를 작성할 수 있게 도와주죠.


Reporting yielded results of long-running Celery task
    https://stackoverflow.com/questions/17052291/reporting-yielded-results-of-long-running-celery-task
https://core-research-team.github.io/2020-03-01/Celery-Flask-30e28a8974974f6cb55ed0c07d042671
https://github.com/runozo/flask-socketio-video-stream/tree/master/livestream
https://github.com/poonesh/Flask-SocketIO-Celery-example
    legacy

https://docs.celeryproject.org/en/stable/userguide/calling.html
Quick Cheat Sheet
    T.delay(arg, kwarg=value)
        Star arguments shortcut to .apply_async. (.delay(*args, **kwargs) calls .apply_async(args, kwargs)).
https://docs.celeryproject.org/en/stable/reference/celery.result.html#celery.result.AsyncResult.get
    Warning
        Waiting for tasks within a task may lead to deadlocks. Please read Avoid launching synchronous subtasks.
    Warning
        Backends use resources to store and transmit results. To ensure that resources are released, you must eventually call get() or forget() on EVERY AsyncResult instance returned after calling a task.
Avoid launching synchronous subtasks
    https://docs.celeryproject.org/en/stable/userguide/tasks.html#task-synchronous-subtasks
    Having a task wait for the result of another task is really inefficient, and may even cause a deadlock if the worker pool is exhausted.

Proxies
    https://docs.celeryproject.org/en/stable/reference/celery.html#proxies
    celery.current_app
        The currently set app for this thread.
    celery.current_task
        The task currently being executed (only set in the worker, or when eager/apply is used).
.s()
    https://docs.celeryproject.org/en/stable/reference/celery.html#celery.signature

https://socket.io/docs/v4/namespaces/


"""
