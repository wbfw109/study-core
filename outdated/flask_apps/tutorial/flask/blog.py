from flask import Blueprint, flash, g, redirect, render_template, request, url_for
from werkzeug.exceptions import abort

from mysite.tutorial.auth import login_required
from mysite.tutorial.db import get_db

bp = Blueprint("tutorial/blog", __name__, url_prefix="/tutorial/blog")


@bp.route("/")
def index():
    db = get_db()
    posts = db.execute(
        "SELECT p.id, title, body, created, author_id, username"
        " FROM post p JOIN user u ON p.author_id = u.id"
        " ORDER BY created DESC"
    ).fetchall()
    return render_template("tutorial/blog/index.html", posts=posts)


@bp.route("/create", methods=("GET", "POST"))
@login_required
def create():
    """
    The create view works the same as the auth register view.
        Either the form is displayed, or the posted data is validated and the post is added to the database or an error is shown.
    ■ The login_required decorator you wrote earlier is used on the blog views.
        A user must be logged in to visit these views, otherwise they will be redirected to the login page.
    """
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        error = None

        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "INSERT INTO post (title, body, author_id)" " VALUES (?, ?, ?)",
                (title, body, g.user["id"]),
            )
            db.commit()
            return redirect(url_for("tutorial/blog.index"))

    return render_template("tutorial/blog/create.html")


def get_post(id, check_author=True):
    """
    Both the update and delete views will need to fetch a post by id and check if the author matches the logged in user.
    To avoid duplicating code, you can write a function to get the post and call it from each view.

    abort() will raise a special exception that returns an HTTP status code.
        It takes an optional message to show with the error, otherwise a default message is used. 404 means “Not Found”, and 403 means “Forbidden”. (401 means “Unauthorized”, but you redirect to the login page instead of returning that status.)

    ● The check_author argument is defined so that the function can be used to get a post without checking the author.
        This would be useful if you wrote a view to show an individual post on a page
        , where the user doesn’t matter because they’re not modifying the post.
    """
    post = (
        get_db()
        .execute(
            "SELECT p.id, title, body, created, author_id, username"
            " FROM post p JOIN user u ON p.author_id = u.id"
            " WHERE p.id = ?",
            (id,),
        )
        .fetchone()
    )

    if post is None:
        abort(404, "Post id {0} doesn't exist.".format(id))

    if check_author and post["author_id"] != g.user["id"]:
        abort(403)

    return post


@bp.route("/<int:id>/update", methods=("GET", "POST"))
@login_required
def update(id):
    post = get_post(id)

    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        error = None

        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "UPDATE post SET title = ?, body = ?" " WHERE id = ?", (title, body, id)
            )
            db.commit()
            return redirect(url_for("tutorial/blog.index"))

    return render_template("tutorial/blog/update.html", post=post)


@bp.route("/<int:id>/delete", methods=("POST",))
@login_required
def delete(id):
    """
    The delete view doesn’t have its own template
    , the delete button is part of update.html and posts to the /<id>/delete URL.
    Since there is no template, it will only handle the POST method and then redirect to the index view.
    """
    get_post(id)
    db = get_db()
    db.execute("DELETE FROM post WHERE id = ?", (id,))
    db.commit()
    return redirect(url_for("tutorial/blog.index"))
