import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce logs
os.environ['KMP_AFFINITY'] = 'disabled'    # Helps in some cases
os.environ['TF_XLA_FLAGS'] = ''            # Disable XLA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only if GPU causes issues
import cv2
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    """
    Load RGB images from a file and validate.
    """
    image_path = os.path.join(data_dir, image_file.strip())
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = mpimg.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Failed to load image or image is empty: {image_path}")
    print(f"Loaded image {image_path} with shape: {image.shape}")  # Debug
    return image

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom).
    """
    if image.shape[0] < 85:  # Ensure image is tall enough for cropping
        print(f"Warning: Image height {image.shape[0]} too small for cropping")
        return image
    print(f"Before crop shape: {image.shape}")  # Debug
    cropped = image[60:-25, :, :]
    print(f"After crop shape: {cropped.shape}")  # Debug
    if cropped.size == 0:
        raise ValueError("Cropped image is empty")
    return cropped

def resize(image):
    """
    Resize the image to the input shape used by the network model.
    """
    print(f"Before resize shape: {image.shape}")  # Debug
    if image.size == 0:
        raise ValueError("Input image to resize is empty")
    resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    print(f"After resize shape: {resized.shape}")  # Debug
    return resized

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does).
    """
    print(f"Before RGB2YUV shape: {image.shape}")  # Debug
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    print(f"After RGB2YUV shape: {yuv.shape}")  # Debug
    return yuv

def preprocess(image):
    """
    Combine all preprocess functions into one.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image to preprocess is empty")
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left, or right, and adjust the steering angle.
    """
    for img_path in [center, left, right]:
        if not img_path or not isinstance(img_path, str):
            raise ValueError(f"Invalid image path: {img_path}")
        full_path = os.path.join(data_dir, img_path.strip())
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")
    
    choice = np.random.choice(3)
    if choice == 0:
        print(f"Choosing left image: {left}, steering adjustment: +0.2")  # Debug
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        print(f"Choosing right image: {right}, steering adjustment: -0.2")  # Debug
        return load_image(data_dir, right), steering_angle - 0.2
    print(f"Choosing center image: {center}, no steering adjustment")  # Debug
    return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow.
    """
    print(f"random_shadow input shape: {image.shape}")  # Debug
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)



def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image given image paths and associated steering angles.
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            try:
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augment(data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                images[i] = preprocess(image)
                steers[i] = steering_angle
                i += 1
            except Exception as e:
                print(f"Error processing image {center}: {e}")
                continue  # Skip invalid images
            if i == batch_size:
                break
        yield images, steers