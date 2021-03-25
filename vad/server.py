from vad import VAD
import numpy as np
import flask
import redis
import uuid
import time
import json
import io
import logging
from flask import Response
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
CORS(app)

detector = VAD(frame_duration = 1, model_path = 'models/vad')
SAMPLING_RATE = 44100

@app.route("/")
def homepage():
    return "Welcome to the REST API!"

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    result = {"success": False}
    frames = flask.request.data
    array_frames = np.frombuffer(frames,dtype=np.int16)
    array_frames = array_frames.astype(np.float32, order='C') / 32768.0
    result['result'] = detector.predict(array_frames,SAMPLING_RATE)
    return flask.jsonify(result)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host='127.0.0.1', port=5700,debug=False)
