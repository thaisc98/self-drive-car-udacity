import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import eventlet
import eventlet.websocket
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import utils
import json

# For Mac M4
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'STRICT'
os.environ['DNNL_VERBOSE'] = '1'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['TF_XLA_FLAGS'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = None
prev_image_array = None
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED
args = None

def process_telemetry(data):
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    if args and args.image_folder != '':
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(args.image_folder, timestamp)
        image.save('{}.jpg'.format(image_filename))
    image = np.asarray(image)
    image = utils.preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image, batch_size=1))
    global speed_limit
    if speed > speed_limit:
        speed_limit = MIN_SPEED
    else:
        speed_limit = MAX_SPEED
    throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
    print('{} {} {}'.format(steering_angle, throttle, speed))
    return steering_angle, throttle

@eventlet.websocket.WebSocketWSGI
def handle(ws):
    sid = "simulator_sid"
    ws.send('0{"sid":"%s","upgrades":[],"pingInterval":25000,"pingTimeout":20000,"maxPayload":1000000}' % sid)
    while True:
        msg = ws.wait()
        if msg is None:
            break
        if msg == '2':
            ws.send('3')
        elif msg == '40':
            ws.send('40')
            # Send initial control
            data = json.dumps(["steer", {'steering_angle': '0', 'throttle': '0'}])
            ws.send('42' + data)
        elif msg.startswith('42'):
            json_str = msg[2:]
            event_data = json.loads(json_str)
            event = event_data[0]
            data = event_data[1] if len(event_data) > 1 else {}
            if event == "telemetry":
                steering_angle, throttle = process_telemetry(data)
                data = json.dumps(["steer", {'steering_angle': str(steering_angle), 'throttle': str(throttle)}])
                ws.send('42' + data)
        else:
            print("Unknown message:", msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model keras file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Start WebSocket server
    listener = eventlet.listen(('', 4567))
    print("Starting WebSocket server...")
    eventlet.wsgi.server(listener, handle)