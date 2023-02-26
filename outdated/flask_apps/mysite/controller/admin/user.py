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

admin_user_bp = Blueprint("admin_user", __name__, url_prefix="/admin/user")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def get_allowed_file(filename: str) -> dict:
    filename_list: list = secure_filename(filename).lower().rsplit(".", 1)
    is_allowed: bool = True
    filename_dict: dict = dict()

    if "." in filename and filename_list[1] in ALLOWED_EXTENSIONS:
        filename_dict["basename"] = filename_list[0]
        filename_dict["extension"] = filename_list[1]
    else:
        is_allowed = False

    filename_dict["is_allowed"] = is_allowed

    return filename_dict


@admin_user_bp.route("/get_thumbnail/<user_id>", methods=("POST",))
def get_thumbnail(user_id) -> dict:
    encoded_string: str = ""
    if request.method == "POST":
        you: User = db_session.query(User).get(user_id)

        import base64

        with open(
            str(current_app.config["UPLOAD_FOLDER"] / you.image), "rb"
        ) as image_file:
            encoded_string = base64.encodebytes(image_file.read()).decode("utf-8")

    return {"image": encoded_string}


@admin_user_bp.route("/")
def index():
    if "txt" in request.args.keys() and request.args["txt"] == "o":
        current_app.logger.debug("data from txt file")
        # * load users from text file
        user_path_from_txt_file: PurePath = (
            current_app.config["RESOURCE_FOLDER"] / "users.txt"
        )
        users: list = []
        with open(str(user_path_from_txt_file), "r", encoding="utf-8") as txt_file:
            for line in txt_file:
                data = line.split("\t")
                users.append(
                    {"id": data[0], "name": data[1], "age": data[2], "image": data[3]}
                )
    else:
        users: list = db_session.query(User).all()

    return render_template("admin/user/index.html", users=users)


@admin_user_bp.route("/create/<user_id>", methods=("GET",))
@admin_user_bp.route("/create/", methods=("GET", "POST"))
def create(user_id=None):
    if user_id == None:
        if request.method == "POST":
            name: str = " ".join(escape(request.form["name"]).split())
            age: int = int(request.form["age"])
            is_error: bool = False
            is_valid_file: bool = False

            if not name:
                is_error = True
                flash("Name is required.")
            elif "image" not in request.files:
                # check if the post request has the file part
                is_error = True
                flash("No file part.")

            image: FileStorage = request.files["image"]
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if image.filename == "" or not image:
                # error = 'No selected file'
                pass
            else:
                filename_dict: dict = get_allowed_file(image.filename)
                if filename_dict["is_allowed"]:
                    image_hash = hashlib.sha256(image.stream.read())
                    image.seek(0)
                    digest = image_hash.hexdigest()

                    filename = "{basename}_{digest}.{extension}".format(
                        basename=filename_dict["basename"],
                        digest=digest,
                        extension=filename_dict["extension"],
                    )

                    is_valid_file = True
                else:
                    is_error = True
                    flash("file is None or File extension is not for picture.")

            if not is_error:
                user: User = User(name, age, "")
                db_session.add(user)
                db_session.flush()
                db_session.refresh(user)

                if is_valid_file:
                    user_path: PurePath = current_app.config["UPLOAD_FOLDER"] / str(
                        user.id
                    )

                    Path(user_path).mkdir(exist_ok=True)
                    user.image = str(PurePath(str(user.id)) / filename)
                    image.save(str(user_path / filename))

                # * create users from text file
                user_path_from_txt_file: PurePath = (
                    current_app.config["RESOURCE_FOLDER"] / "users.txt"
                )

                if not Path(user_path_from_txt_file).exists():
                    with open(
                        user_path_from_txt_file, "w", encoding="utf-8"
                    ) as txt_file:
                        pass

                with open(
                    str(user_path_from_txt_file), "a", encoding="utf-8"
                ) as txt_file:
                    txt_file.write(
                        "{data}\n".format(
                            data="\t".join(
                                [str(user.id), user.name, str(user.age), user.image]
                            )
                        )
                    )

                db_session.commit()

            return redirect(url_for("admin_user.index"))
    else:
        if request.method == "GET":
            you: User = db_session.query(User).get(user_id)
        return render_template("admin/user/create.html", you=you)

    return render_template("admin/user/create.html", you=None)


# def get_post(id, check_author=True):
#     post = (
#         get_db()
#         .execute(
#             "SELECT p.id, title, body, created, author_id, username"
#             " FROM post p JOIN user u ON p.author_id = u.id"
#             " WHERE p.id = ?",
#             (id,),
#         )
#         .fetchone()
#     )

#     if post is None:
#         abort(404, "Post id {0} doesn't exist.".format(id))

#     if check_author and post["author_id"] != g.user["id"]:
#         abort(403)

#     return post

# @user_bp.route("/<int:id>/update", methods=("GET", "POST"))
# def update(id):
#     post = get_post(id)

#     if request.method == "POST":
#         title = request.form["title"]
#         body = request.form["body"]
#         error = None

#         if not title:
#             error = "Title is required."

#         if error is not None:
#             flash(error)
#         else:
#             db = get_db()
#             db.execute(
#                 "UPDATE post SET title = ?, body = ?" " WHERE id = ?", (title, body, id)
#             )
#             db.commit()
#             return redirect(url_for("user.index"))
#     return render_template("user/update.html", post=post)

# @user_bp.route('/<int:id>/delete', methods=('POST',))
# def delete(id):
#     """
#     The delete view doesn’t have its own template
#     , the delete button is part of update.html and posts to the /<id>/delete URL.
#     Since there is no template, it will only handle the POST method and then redirect to the index view.
#     """
#     get_post(id)
#     db = get_db()
#     db.execute('DELETE FROM post WHERE id = ?', (id,))
#     db.commit()
#     return redirect(url_for('user.index'))

# @flask_app.route("/test_login")
# def test_login():
#     if "username" in session:
#         flash("You already were successfully logged in")
#     else:
#         # remove the username from the session if it's there
#         session["username"] = "imuser"
#     return redirect(url_for("index"))

# @flask_app.route("/test_logout")
# def test_logout():
#     session.pop("username", None)
#     return redirect(url_for("index"))

# @flask_app.route("/me")
# def me_api():
#     user = get_current_user()
#     return {
#         "username": user.username,
#         "theme": user.theme,
#         "image": url_for("user_image", filename=user.image),
#     }

# @flask_app.route("/users")
# def users_api():
#     users = get_all_users()
#     return jsonify([user.to_json() for user in users])

# @flask_app.route("/login", methods=["POST", "GET"])
# def login():
#     error = None
#     if request.method == "POST":
#         if valid_login(request.form["username"], request.form["password"]):
#             return log_the_user_in(request.form["username"])
#         else:
#             error = "Invalid username/password"

#     # the code below is executed if the request method
#     # was GET or the credentials were invalid
#     return render_template("login.html", error=error)

# @flask_app.route("/string_view")
# def string_view():
#     if "username" in session:
#         return "Logged in as %s" % escape(session["username"])
#     else:
#         return "This is string_view"

# @flask_app.route("/redirect_and_error")
# def redirect_and_error():
#     # Storing cookies:
#     resp = make_response(render_template("error.html"), 404)
#     resp.headers["X-Something"] = "A value"
#     resp.set_cookie("username", "usernameB")
#     # Reading cookies:
#     # show the user profile for that user
#     username = request.cookies.get("username")
#     # use cookies.get(key) instead of cookies[key] to not get a KeyError if the cookie is missing.
#     print(username)

#     if username == "the username":
#         return redirect(url_for("profile", username="outsider"))
#     return resp

# @flask_app.route("/user/<username>")
# def profile(username):
#     if username == "outsider":
#         abort(401)
#     return "{}'s profile".format(escape(username))

# # is possible to use relative path?

# @flask_app.route("/post/<int:post_id>")
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return "Post %d" % post_id

# @flask_app.route("/path/<path:subpath>")
# def show_subpath(subpath):
#     # show the subpath after /path/
#     return "Subpath %s" % escape(subpath)

# # @app.errorhandler(404)
# # def page_not_found(error):
# #     return render_template("page_not_found.html"), 404

# Markup("<strong>Hello %s!</strong>") % "<blink>hacker</blink>"
# Markup.escape("<blink>hacker</blink>")
# Markup("<em>Marked up</em> &raquo; HTML").striptags()
