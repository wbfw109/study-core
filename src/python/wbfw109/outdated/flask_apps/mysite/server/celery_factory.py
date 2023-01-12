"""celery function using factory pattern to avoid circular
"""
from mysite import config
from celery import Celery

# from celery.app.task import Task
from celery.app.task import Task
from flask import Flask

celery_default = Celery("celery_example", include=["backend.server.celery_tasks"])


def configure_celery(app: Flask) -> Celery:
    TaskBase: Task = celery_default.Task

    # Initialization of instance is not here anymore
    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    # Configuration of placeholder happens here
    celery_default.conf.update(
        broker_url=app.config["CELERY_BROKER_URL"],
        result_backend=app.config["CELERY_RESULT_BACKEND"]
        # Rest of configuration
    )
    celery_default.Task = ContextTask
    return celery_default
