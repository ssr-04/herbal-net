
from flask import Flask, render_template, request, redirect,jsonify,session
from model import run_model,get_details

app = Flask(__name__)

@app.after_request
def after_request(response):
  """Ensure responses aren't cached"""
  response.headers["Cache-Control"] = "no-cache,no-store,must-revalidate"
  response.headers["Expires"] = 0
  response.headers["Pragma"] = "no-cache"
  return response



@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/magic",methods=["POST"])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            return "Illegal Post detected 🤨"
        file = request.files['file']

        if file.filename == '':
            return "Invalid","Choose file before submitting"

        if file:
            filename = (file.filename).split('.')
            model = int(request.form.get("modelSelect"))
            path = "./static/predcit_image"+ '.' + filename[-1]
            file.save(path)
            print("started")
            out = run_model(model,path)
            detail = get_details(out[0])
            print("done")
            return render_template("result.html",path = filename[-1], label=out[0], accuracy=round(out[1],2), description = detail)

        return "Something went wrong"


if (__name__ == "__main__"):
    print("started")
    app.run(host="0.0.0.0", port = 9090, debug= True)