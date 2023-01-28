import sqlite3
from pathlib import Path
import click
from flask import current_app, g
from flask.cli import with_appcontext


# ~ Connect to the Database
def get_db():
    """
    g is a special object that is unique for each request. It is used to store data that might be accessed by multiple functions during the request.
    The connection is stored and reused instead of creating a new connection if get_db is called a second time in the same request.

    current_app is another special object that points to the Flask application handling the request. Since you used an application factory, there is no application object when writing the rest of your code. get_db will be called when the application has been created and is handling a request, so current_app can be used.

    sqlite3.connect() establishes a connection to the file pointed at by the DATABASE configuration key. This file doesn’t have to exist yet, and won’t until you initialize the database later.

    sqlite3.Row tells the connection to return rows that behave like dicts. This allows accessing the columns by name.

    close_db checks if a connection was created by checking if g.db was set. If the connection exists, it is closed. Further down you will tell your application about the close_db function in the application factory so that it is called after each request.
    """
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop("db", None)

    if db is not None:
        db.close()


# ~ Create the Tables
def init_db():
    """
    open_resource() opens a file relative to the tutorial package, which is useful since you won’t necessarily know where that location is when deploying the application later. get_db returns a database connection, which is used to execute the commands read from the file.
    """
    db = get_db()

    with current_app.open_resource(Path("resource/tutorial_schema.sql")) as f:
        db.executescript(f.read().decode("utf8"))


@click.command("init-db")
@with_appcontext
def init_db_command():
    """
    click.command() defines a command line command called init-db that calls the init_db function and shows a success message to the user. You can read Command Line Interface to learn more about writing commands.
    Clear the existing data and create new tables.
    https://flask.palletsprojects.com/en/1.1.x/cli/#cli
    """
    init_db()
    click.echo("Initialized the database.")


# ~ Register with the Application
def init_app(app):
    """
    The close_db and init_db_command functions need to be registered with the application instance; otherwise, they won’t be used by the application. However, since you’re using a factory function, that instance isn’t available when writing the functions. Instead, write a function that takes an application and does the registration.

    app.teardown_appcontext() tells Flask to call that function when cleaning up after returning the response.

    app.cli.add_command() adds a new command that can be called with the flask command.
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
