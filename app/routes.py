from flask import request, render_template, jsonify
from . import application
import json
from app.ari import relation_identification


@application.route("/", methods=["GET", "POST"])
def amf_ari():
    if request.method == "POST":
        f = request.files["file"]
        f.save(f.filename)
        with open(f.filename, "r") as ff:
            content = json.load(ff)
        # Predict existing relations in content (i.e., xaif file) "I" nodes.
        window_size_param = request.args.get("window_size")
        if window_size_param is None:
            window_size = -1
        else:
            try:
                window_size = int(window_size_param)
            except (TypeError, ValueError):
                return jsonify({"error": "window_size must be an integer"}), 400

            if window_size != -1 and window_size < 2:
                return jsonify({"error": "window_size must be -1 or >= 2"}), 400

        response = relation_identification(content, window_size=window_size)
        print(response)
        return jsonify(response)
    elif request.method == "GET":
        return render_template("docs.html")
