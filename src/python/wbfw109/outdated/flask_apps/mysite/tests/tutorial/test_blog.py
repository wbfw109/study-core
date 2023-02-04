import pytest
from mysite.tutorial.db import get_db

"""
A user must be logged in to access the create, update, and delete views.
The logged in user must be the author of the post to access update and delete
, otherwise a 403 Forbidden status is returned. If a post with the given id doesn’t exist
, update and delete should return 404 Not Found.
"""


def test_index(client, auth):
    response = client.get("/tutorial/")
    assert b"Log In" in response.data
    assert b"Register" in response.data

    auth.login()
    response = client.get("/tutorial/")
    assert b"Log Out" in response.data
    assert b"test title" in response.data
    assert b"by test on 2018-01-01" in response.data
    assert b"test\nbody" in response.data
    assert b'href="/tutorial/blog/1/update"' in response.data


"""
The create and update views should render and return a 200 OK status for a GET request.
When valid data is sent in a POST request, create should insert the new post data into the database
, and update should modify the existing data. Both pages should show an error message on invalid data.
"""


@pytest.mark.parametrize(
    "path",
    (
        "/tutorial/blog/create",
        "/tutorial/blog/1/update",
        "/tutorial/blog/1/delete",
    ),
)
def test_login_required(client, path):
    response = client.post(path)
    assert response.headers["Location"] == "http://localhost/tutorial/auth/login"


def test_author_required(app, client, auth):
    # change the post author to another user
    with app.app_context():
        db = get_db()
        db.execute("UPDATE post SET author_id = 2 WHERE id = 1")
        db.commit()

    auth.login()
    # current user can't modify other user's post
    assert client.post("/tutorial/blog/1/update").status_code == 403
    assert client.post("/tutorial/blog/1/delete").status_code == 403
    # current user doesn't see edit link
    assert b'href="/tutorial/blog/1/update"' not in client.get("/tutorial/").data


@pytest.mark.parametrize(
    "path",
    (
        "/tutorial/blog/2/update",
        "/tutorial/blog/2/delete",
    ),
)
def test_exists_required(client, auth, path):
    auth.login()
    assert client.post(path).status_code == 404


def test_create(client, auth, app):
    auth.login()
    assert client.get("/tutorial/blog/create").status_code == 200
    client.post("/tutorial/blog/create", data={"title": "created", "body": ""})

    with app.app_context():
        db = get_db()
        count = db.execute("SELECT COUNT(id) FROM post").fetchone()[0]
        assert count == 2


def test_update(client, auth, app):
    auth.login()
    assert client.get("/tutorial/blog/1/update").status_code == 200
    client.post("/tutorial/blog/1/update", data={"title": "updated", "body": ""})

    with app.app_context():
        db = get_db()
        post = db.execute("SELECT * FROM post WHERE id = 1").fetchone()
        assert post["title"] == "updated"


@pytest.mark.parametrize(
    "path",
    (
        "/tutorial/blog/create",
        "/tutorial/blog/1/update",
    ),
)
def test_create_update_validate(client, auth, path):
    auth.login()
    response = client.post(path, data={"title": "", "body": ""})
    assert b"Title is required." in response.data


"""
The delete view should redirect to the index URL and the post should no longer exist in the database.
"""


def test_delete(client, auth, app):
    auth.login()
    response = client.post("/tutorial/blog/1/delete")
    assert response.headers["Location"] == "http://localhost/tutorial/blog/"

    with app.app_context():
        db = get_db()
        post = db.execute("SELECT * FROM post WHERE id = 1").fetchone()
        assert post is None
