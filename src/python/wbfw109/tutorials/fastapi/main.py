from enum import Enum

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


# Title: (Data conversion, Data validation)
@app.get("/items/{item_id}")
async def read_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


# Title: Order matters
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}


# Title:  Path parameters
# By inheriting from str the API docs will be able to know that the values must be of type string and will be able to render correctly.
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# You can return enum members from your path operation, even nested in a JSON body (e.g. a dict).
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


# Path parameters containing paths
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    """You could need the parameter to contain /home/johndoe/myfile.txt, with a leading slash (/).
    In that case, the URL would be: /files//home/johndoe/myfile.txt, with a double slash (//) between files and home
    """
    return {"file_path": file_path}
