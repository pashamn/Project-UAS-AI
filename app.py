from flask import Flask, render_template, request
from predict import predict_image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    label, confidence = predict_image(file_path)

    return render_template("result.html",
                           label=label,
                           confidence=round(confidence, 2),
                           image_name=filename)

if __name__ == "__main__":
    app.run(debug=True)
