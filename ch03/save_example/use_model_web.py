# 1. Import the necessary libraries:
import flask
from flask import request
import torch
import final_model

# 2. Initialize the Flask app:
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# The DEBUG configuration is set to True during development but should be set
# to False when in production.
# 3. Load a previously trained model:
def load_model_checkpoint(path):
    checkpoint = torch.load(path)
    model = final_model.Classifier(checkpoint['input'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_model_checkpoint('checkpoint.pth')

# 4. Define the route where the API will be accessible, as well as the method(s) that
# can be used for sending information to the API in order to perform an action.
# This syntax is called a decorator and should be located immediately before
# a function:

# 5.Define a function that performs the desired action. In this case, the function will
# take the information that was sent to the API and feed it to the previously loaded
# model to perform a prediction. Once the prediction has been obtained, the
# function should return a response, which will be displayed as the result of the
# API request:
@app.route('/prediction', methods=['POST'])
def prediction():

    body = request.get_json()
    example = torch.tensor(body['data']).float()
    pred = model(example)
    pred = torch.exp(pred)
    _, top_class_test = pred.topk(1, dim=1)
    top_class_test = top_class_test.numpy()

    return {"status": "ok", "result": int(top_class_test[0][0])}

app.run(debug=True, use_reloader=False)


