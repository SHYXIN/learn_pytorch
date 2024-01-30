# Using Flask, we will create a web API that will receive some data when it's called and
# will return a piece of text that will be displayed in a browser. Follow these steps to
# complete this exercise

# import the required libraries:
import flask
from flask import request
# 2. Initialize the Flask app:


app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Define the route of your API so that it's /<name>. Set the method to GET. Next,
# define a function that takes in an argument (name) and returns a string that
# contains an h1 tag with the word HELLO, followed by the argument received by
# the function:


@app.route('/<name>', methods=['GET'])
def hello(name):
    return "<h1>HELLO {}<h1>".format(name.upper())


# 4. Run the Flask app:
app.run(debug=True, use_reloader=True)

