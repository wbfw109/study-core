"""module for application factory pattern.

when you want to open Flask server with Werkzeug,
    $bash export FLASK_APP=backend FLASK_ENV=development; flask run
        current sever host: http://172.26.180.165:5000/
"""
from mysite.config import CONFIG_CLASS
from pathlib import Path
from logging import debug
import sys
from flask import (
    Flask,
    url_for,
    request,
    render_template,
    make_response,
    abort,
    redirect,
    session,
    flash,
    send_from_directory,
)


def create_app() -> Flask:
    """Function using application factory pattern.

    Returns:
        Flask: [description]
    """
    app = Flask(
        __name__,
        static_folder=f"{Path('..') / CONFIG_CLASS.STATIC_FOLDER}",
        template_folder=f"{Path('..') / CONFIG_CLASS.TEMPLATE_FOLDER}",
        instance_relative_config=True,
    )

    # * routing setting Start
    # from mysite.controller.admin import admin_user_bp
    from mysite.controller.admin.index import admin_bp
    from mysite.controller.admin.user import admin_user_bp
    from mysite.controller.admin.ml import admin_ml_bp

    app.register_blueprint(admin_bp)
    app.register_blueprint(admin_user_bp)
    app.register_blueprint(admin_ml_bp)

    from mysite.controller.stream.streaming import streaming_bp

    app.register_blueprint(streaming_bp)

    @app.route("/")
    def default_page():
        return redirect(url_for("admin_user.index"))

    @app.route("/uploads/<path:subpath>")
    def download_file(subpath):
        return send_from_directory(
            app.config["UPLOAD_FOLDER"], subpath, as_attachment=True
        )

    print("app url map: {url_map}".format(url_map=app.url_map))
    # * routing setting End

    # * database setting Start
    from mysite.database import db_session, init_db

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

    # when app starts, create tables
    init_db()
    # * database setting End

    # # * set config file Start
    app.config.from_object(CONFIG_CLASS)
    CONFIG_CLASS.ensure_folders_exist()
    # * set config file End

    # * rest setting Start
    # redis_client = FlaskRedis(app)
    # * rest setting End

    # ~ code for tutorial
    # # ensure the instance folder exists
    # try:
    #     os.makedirs(app.instance_path)
    # except OSError:
    #     pass
    # # a simple page that says hello
    # @app.route("/hello_world")
    # def hello_world():
    #     return "Hello, World!"
    # @app.route("/hello/")
    # @app.route("/hello/<name>")
    # def hello(name=None):
    #     # with app.test_request_context():
    #     #     print(url_for("index"))
    #     #     print(url_for("login"))
    #     #     # It accepts the name of the function as its first argument and any number of keyword arguments, each corresponding to a variable part of the URL rule.
    #     #     # Unknown variable parts are appended to the URL as query parameters.
    #     #     print(url_for("login", next="/"))
    #     #     print(url_for("profile", username="John Doe"))

    #     with app.test_request_context("/hello", method="POST"):
    #         # now you can do something with the request until the
    #         # end of the with block, such as basic assertions:
    #         assert request.path == "/hello"
    #         assert request.method == "POST"

    #     return render_template("hello.html", name=name)

    # from mysite.tutorial import db
    # db.init_app(app)

    # Import and register the blueprint from the factory using app.register_blueprint(). Place the new code at the end of the factory function before returning the app.
    # from mysite.tutorial import auth
    # app.register_blueprint(auth.bp)

    # # Import and register the blueprint from the factory using app.register_blueprint(). Place the new code at the end of the factory function before returning the app.
    # from mysite.tutorial import blog
    # app.register_blueprint(blog.bp)

    # """
    # However, the endpoint for the index view defined below will be blog.index.
    # Some of the authentication views referred to a plain index endpoint.
    # app.add_url_rule() associates the endpoint name 'index' with the / url so that url_for('index') or url_for('blog.index') will both work
    #   , generating the same / URL either way.

    # In another application you might give the blog blueprint a url_prefix and define a separate index view in the application factory
    #   , similar to the hello view. Then the index and blog.index endpoints and URLs would be different.
    # """
    # app.add_url_rule('/tutorial/', endpoint='tutorial/blog.index')
    #
    return app
