from flask import Flask, render_template, request
import numpy as np
from methods.gaussian import parse_input, gauss_elim

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        eq1 = request.form["eq1"]
        eq2 = request.form["eq2"]
        eq3 = request.form["eq3"]

        A = []
        B = []

        for eq in [eq1, eq2, eq3]:
            coeffs, constant = parse_input(eq)
            A.append(coeffs)
            B.append([constant])

        A = np.array(A, dtype=object)
        B = np.array(B, dtype=object)

        result, steps = gauss_elim(A, B)

        variables = ["x", "y", "z"]  # <-- IMPORTANT

        return render_template("index.html", result=result, steps=steps, variables=variables)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)