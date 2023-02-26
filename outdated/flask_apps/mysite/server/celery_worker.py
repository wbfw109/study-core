"""
:data:`celery` worker instance

if you use celery, require follwing command for celery server in new python environment activated shell
    celery -A backend.server.celery_worker.celery worker -P gevent --loglevel=info
if you remove pending messages, type this.
    celery -A backend.server.celery_worker.celery purge
"""
from mysite import create_app
from mysite.server.celery_factory import configure_celery

celery = configure_celery(create_app())
