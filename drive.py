import os
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
import utils

# Environment settings for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
os.environ['DNNL_VERBOSE'] = '1'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['TF_XLA_FLAGS'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize SocketIO server with explicit async mode
sio = socketio.Server(logger=True, engineio_logger=True, async_mode='eventlet')
app = Flask(__name__)
app = socketio.WSGIApp(sio, app)

# Global variables
model = None
prev_image_array = None
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

@sio.on('connect', namespace='/')
def connect(sid, environ):
    """Handle client connection."""
    print(f"Connected to simulator {sid}, environ: {environ}")
    send_control(0, 0, sid)

@sio.on('telemetry', namespace='/')
def telemetry(sid, data):
    """Handle telemetry data from the simulator."""
    if sid is None:
        print("Error: Received telemetry with sid=None, skipping")
        return
    if not data:
        print(f"No telemetry data received from {sid}")
        sio.emit('manual', data={}, to=sid, namespace='/')
        return

    print(f"Received telemetry from {sid}: {data.keys()}")
    try:
        # Extract telemetry data
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        print(f"Received image size: {image.size}, mode: {image.mode}")

        # Save frame if image folder is specified
        if args.image_folder:
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save(f'{image_filename}.jpg')

        # Preprocess image
        image = np.asarray(image)
        print(f"Image shape before preprocess: {image.shape}")
        image = utils.preprocess(image)
        print(f"Image shape after preprocess: {image.shape}, min: {image.min()}, max: {image.max()}")
        image = np.array([image])

        # Predict steering angle
        steering_angle = float(model.predict(image, verbose=0)[0])
        print(f"Predicted steering angle: {steering_angle}")

        # Adjust throttle based on speed
        global speed_limit
        if speed > speed_limit:
            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED
        throttle = max(0.0, min(1.0, 1.0 - steering_angle**2 - (speed/speed_limit)**2))
        print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.4f}")

        send_control(steering_angle, throttle, sid)
    except Exception as e:
        print(f"Error during telemetry processing: {e}")
        raise

def send_control(steering_angle, throttle, sid):
    """Send steering and throttle commands to the simulator."""
    if sid is None:
        print("Error: Cannot send control, sid is None")
        return
    data = {
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }
    print(f"Sending steer event to {sid}: {data}")
    sio.emit('steer', data=data, to=sid, namespace='/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    try:
        model = tf.keras.models.load_model(args.model, compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    if args.image_folder:
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
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)