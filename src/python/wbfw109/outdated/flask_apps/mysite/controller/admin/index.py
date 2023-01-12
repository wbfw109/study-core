from pathlib import PurePath, Path
from flask import (
    Blueprint,
    flash,
    current_app,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from markupsafe import escape
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
from mysite.database import db_session
from mysite.models import User
import hashlib

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/", methods=("GET",))
def index():
    links = []
    for rule in current_app.url_map.iter_rules():
        links.append((rule.endpoint, rule))
    return render_template("admin/index.html", links=sorted(links, key=lambda x: x[0]))
