import os
import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.optimizers import Adam
import utils
import argparse

# Disable XLA to avoid x2APIC errors MAC M4
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

DATA_PATH = "/Users/thaiscontreras/Documents/autopilot_project/self-drive-car-udacity/data"
CSV_FILE = 'driving_log.csv'
BATCH_SIZE = 16
EPOCHS = 30
CORRECTION = 0.3

def load_data(data_path, csv_file):
    """
    Load image paths and steering angles from CSV, apply corrections for left/right images.
    """
    samples = []
    csv_path = os.path.join(data_path, csv_file)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for line in reader:
            center_path, left_path, right_path, steering = line[0], line[1], line[2], float(line[3])
            samples.append((center_path, steering))
            samples.append((left_path, steering + CORRECTION))
            samples.append((right_path, steering - CORRECTION))
    
    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2, random_state=42)
    return train_samples, validation_samples

def generator(samples, batch_size, data_path):
    """
    Generator to load and preprocess images in batches.
    """
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for img_path, angle in batch_samples:
                # Load image
                full_path = os.path.join(data_path, img_path.strip())
                if not os.path.exists(full_path):
                    print(f"Warning: Image not found at {full_path}")
                    continue
                image = cv2.imread(full_path)
                if image is None:
                    print(f"Warning: Failed to load image at {full_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = utils.preprocess(image)
                images.append(image)
                angles.append(angle)
                images.append(cv2.flip(image, 1))
                angles.append(-angle)
            
            X = np.array(images)
            y = np.array(angles)
            yield X, y

def create_model():
    """
    Create NVIDIA-style CNN model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=utils.INPUT_SHAPE),
        tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1) 
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--data_path',
        type=str,
        default=DATA_PATH,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        default=CSV_FILE,
        help='Name of CSV file with driving data'
    )
    args = parser.parse_args()
    print(args)

    print(f"Loading data from {os.path.join(args.data_path, args.csv_file)}")
    train_samples, validation_samples = load_data(args.data_path, args.csv_file)
    print(f"Number of training samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(validation_samples)}")

    train_generator = generator(train_samples, BATCH_SIZE, args.data_path)
    validation_generator = generator(validation_samples, BATCH_SIZE, args.data_path)

    model = create_model()
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "model-{epoch:03d}.keras",
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5,mode='min')

    try:
        model.fit(
            train_generator,
            steps_per_epoch=len(train_samples) // BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=len(validation_samples) // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint,earlyStopping],
            verbose=1
        )
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == '__main__':
    main()