from flask import render_template, Flask
import os

app = Flask(__name__)

print(os.getcwd())


@app.route('/')
def home():
    return render_template('detect.html')


@app.route('/about')
def detail():
    return "Aurthor of this page was to Aaditya chapagain"


if __name__ == "__main__":
    app.run(debug=True)
