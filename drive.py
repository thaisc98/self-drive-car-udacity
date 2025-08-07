import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
os.environ['DNNL_VERBOSE'] = '1'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['TF_XLA_FLAGS'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import base64
from datetime import datetime
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import tensorflow as tf
print(tf.__version__)
import utils

sio = socketio.Server(logger=True, engineio_logger=True)
app = socketio.WSGIApp(sio)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

@sio.on('connect')
def connect(sid, environ):
    print(f"Connected to simulator {sid}, environ: {environ}")
    sio.emit('steer', {'steering_angle': '0.1', 'throttle': '0.5'})

@sio.on('telemetry')
def telemetry(sid, data):
    print(f"Received telemetry from {sid}: {data.keys()}")
    sio.emit('steer', {'steering_angle': '0.1', 'throttle': '0.5'})
    if sid is None:
        print("Error: Received telemetry with sid=None, skipping")
        return
    print(f"Received telemetry from {sid}: {data.keys()}")
    send_control(0.1, 0.5, sid)  # Hardcoded steering and throttle
    if data:
        print(f"Received telemetry from {sid}: {data.keys()}")
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        print(f"Received image size: {image.size}, mode: {image.mode}")
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
        try:
            image = np.asarray(image)
            print(f"Image shape before preprocess: {image.shape}")
            image = utils.preprocess(image)
            print(f"Preprocessed image shape: {image.shape}, min: {image.min()}, max: {image.max()}")
            print(f"Image shape after preprocess: {image.shape}")
            image = np.array([image])
            steering_angle = float(model.predict(image, verbose=0)[0])
            print(f"Predicted steering angle: {steering_angle}")
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = max(0.0, min(1.0, 1.0 - steering_angle**2 - (speed/speed_limit)**2))  # Clamp throttle
            print(f"Calculated throttle: {throttle}")
            print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.4f}")
            send_control(steering_angle, throttle, sid)
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    else:
        print(f"No telemetry data received from {sid}")
#         sio.emit('manual', data={}, to=sid, namespace='/')

def send_control(steering_angle, throttle,sid):
    print(f"Sending control to {sid}: steering={steering_angle}, throttle={throttle}")
    data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
    print(f"Sending steer event to {sid}: {data}")
    sio.emit("steer", data=data, to=sid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--model',
        type=str,
        default='/Users/thaiscontreras/Documents/autopilot_project/self-drive-car-udacity/model-001.keras',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        default='/Users/thaiscontreras/Documents/autopilot_project/self-drive-car-udacity/data/trained',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    try:
        model = tf.keras.models.load_model(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    if args.image_folder != '':
        print(f"Creating image folder at {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    print("Starting WSGI server...")
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)