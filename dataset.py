import tensorflow as tf
import os
import random

IMG_SIZE = 128
BATCH_SIZE = 32

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def high_pass_residual(image):
    blur = tf.nn.avg_pool2d(
        tf.expand_dims(image, 0),
        ksize=5,
        strides=1,
        padding="SAME"
    )
    return image - blur[0]

def random_patch(image):
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    top = tf.random.uniform([], 0, h - IMG_SIZE, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - IMG_SIZE, dtype=tf.int32)

    return image[top:top+IMG_SIZE, left:left+IMG_SIZE, :]

def stego_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    noise = tf.random.normal(tf.shape(image), stddev=0.005)
    image = image + noise

    return tf.clip_by_value(image, -1.0, 1.0)

def process(path, label, training=True):
    image = load_image(path)
    image = high_pass_residual(image)
    image = random_patch(image)

    if training:
        image = stego_augmentation(image)

    return image, label

def build_dataset(base_dir, training=True):
    clean = []
    infected = []

    for root, _, files in os.walk(os.path.join(base_dir, "clean")):
        for f in files:
            clean.append(os.path.join(root, f))

    for root, _, files in os.walk(os.path.join(base_dir, "infected")):
        for f in files:
            infected.append(os.path.join(root, f))

    paths = clean + infected
    labels = [0]*len(clean) + [1]*len(infected)

    combined = list(zip(paths, labels))
    random.shuffle(combined)

    paths, labels = zip(*combined)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(lambda x, y: process(x, y, training),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
