from flask import (
    Blueprint,
    current_app,
    render_template,
)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/", methods=("GET",))
def index():
    links = []
    for rule in current_app.url_map.iter_rules():
        links.append((rule.endpoint, rule))
    return render_template("admin/index.html", links=sorted(links, key=lambda x: x[0]))
