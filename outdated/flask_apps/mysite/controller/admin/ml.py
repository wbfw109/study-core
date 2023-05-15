from mysite.config import CONFIG_CLASS
from pathlib import Path
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.datastructures import FileStorage
import tensorflow as tf
import tensorflow.keras as keras
from keras_adabound import AdaBound
import pandas
import numpy as np
from PIL import Image

admin_ml_bp = Blueprint("admin_ml", __name__, url_prefix="/admin/ml")


@admin_ml_bp.route("/")
def index():
    return redirect(url_for("admin_ml.linear"))


@admin_ml_bp.route("/cnn/", methods=("GET", "POST"))
def cnn():
    cifar_10_model_path: Path = (
        CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab Notebooks/cifar_10/model-CNN-84"
    )

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    if request.method == "POST":
        is_error: bool = False

        image: FileStorage = request.files["image"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if image.filename == "" or not image:
            is_error = True
            flash("No selected file")
            return render_template("admin/ml/cnn.html", class_names=class_names)

        if not is_error:
            image_test_generator = keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
            )
            cifar_10_model: keras.models.Sequential = keras.models.load_model(
                str(cifar_10_model_path), custom_objects={"AdaBound": AdaBound}
            )

            resized_image = (
                Image.open(image).convert("RGB").resize((32, 32), Image.NEAREST)
            )
            image_np = keras.preprocessing.image.img_to_array(resized_image)
            image_np = np.expand_dims(image_np, axis=0)
            image_test_generator.fit(image_np)

            class_predicted = class_names[
                np.argmax(
                    cifar_10_model.predict(image_test_generator.flow(image_np)), axis=-1
                ).item()
            ]

            return render_template(
                "admin/ml/cnn.html",
                class_names=class_names,
                class_predicted=class_predicted,
            )
        else:
            return render_template("admin/ml/cnn.html", class_names=class_names)

    elif request.method == "GET":
        return render_template("admin/ml/cnn.html", class_names=class_names)


@admin_ml_bp.route("/linear/", methods=("GET", "POST"))
def linear():
    red_wine_model_path: Path = (
        CONFIG_CLASS.GOOGLE_DRIVE_APP_PATH / "Colab Notebooks/red_wine/saved_model"
    )

    input_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    filtered_features_copy_literal = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "chlorides",
        "total sulfur dioxide",
        "density",
        "sulphates",
        "alcohol",
    ]
    if request.method == "POST":
        is_error: bool = False

        linear_x: dict = {}

        # else:
        for key in [
            key for key in request.form.keys() if key in filtered_features_copy_literal
        ]:
            if not request.form[key]:
                is_error = True
                flash("not allowed empty values")
                break
            else:
                linear_x[key] = float(request.form[key])

        if not is_error:
            loaded_model = tf.saved_model.load(str(red_wine_model_path))
            input_test_data_df = pandas.DataFrame.from_dict([linear_x])
            infer = loaded_model.signatures["serving_default"]

            result = infer(
                x_n_tf=tf.reshape(
                    tf.convert_to_tensor(input_test_data_df.iloc[0], dtype=tf.float64),
                    shape=(1, input_test_data_df.iloc[0].size),
                )
            )
            print("=======================")
            print(result)
            print("=======================")

            return render_template(
                "admin/ml/linear.html",
                input_columns=input_columns,
                result=list(result.values())[0].numpy().item(),
            )
        else:
            return render_template("admin/ml/linear.html", input_columns=input_columns)

    elif request.method == "GET":
        # test values
        # random_test_data_df = pandas.DataFrame.from_dict(
        #     {
        #         x: [random.random() + random.randint(0, 5)]
        #         for x in filtered_features_copy_literal
        #     }
        # )

        # loaded_model = tf.saved_model.load(str(red_wine_model_path))
        # infer = loaded_model.signatures["serving_default"]

        # result = infer(
        #     x_n_tf=tf.reshape(
        #         tf.convert_to_tensor(random_test_data_df.iloc[0], dtype=tf.float64),
        #         shape=(1, random_test_data_df.iloc[0].size),
        #     )
        # )

        # return render_template(
        #     "admin/ml/linear.html",
        #     input_columns=input_columns,
        #     result=np.asscalar(list(result.values())[0].numpy()),
        # )
        return render_template("admin/ml/linear.html", input_columns=input_columns)
