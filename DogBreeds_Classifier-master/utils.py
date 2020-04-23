import cv2
import os
import numpy as np
from scipy import ndimage
import pandas as pd
import sys
import tqdm
import keras.preprocessing.image

# Collection of methods for data operations. Implemented are functions to read
# images/masks from files and to read basic properties of the train/test
# data sets.


def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)

        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)

    if target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
    return mask


def calculate_weights_from_dir(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    list_of_masks = []
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, None)
        if target_size:
            mask_tmp = cv2.resize(mask_tmp, target_size,
                                  interpolation=cv2.INTER_AREA)
        list_of_masks.append(mask_tmp)
        if not i:
            merged_mask = mask_tmp
        else:
            merged_mask = np.maximum(merged_mask, mask_tmp)

    weights = calculate_weight(
        merged_mask, list_of_masks)  # list is grey

    return weights


def calculate_weight(merged_mask, masks, w0=10, q=5):
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array(
        [ndimage.distance_transform_edt(m == 0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1 + d2)**2 / (2 * q**2)).astype(np.float32)
    weight = (merged_mask == 0) * weight
    return weight


def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(train_dir))[1]):

        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name)
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir])

    train_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                          'img_ratio', 'num_channels',
                                          'num_masks', 'image_path',
                                          'mask_dir'])
    return train_df


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):

        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels',
                                         'image_path'])
    return test_df


def imshow_args(x):
    """Matplotlib imshow arguments for plotting."""
    if len(x.shape) == 2:
        return x
    if x.shape[2] == 1:
        return x[:, :, 0]
    elif x.shape[2] == 3:
        return x


def load_raw_data(train_df, image_size=(256, 256)):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, y_weights = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']),
                                 total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=image_size)
        mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
        weights = calculate_weights_from_dir(train_df['mask_dir'].loc[i],
                                             target_size=image_size)
        x_train.append(img)
        y_train.append(mask)
        y_weights.append(weights)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    y_train = np.expand_dims(np.array(y_train), axis=3)
    y_weights = np.expand_dims(np.array(y_weights), axis=3)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, x_train.dtype))
    print('y_weights.shape: {} of dtype {}'.format(
        y_weights.shape, y_weights.dtype))

    return x_train, y_train, y_weights


def load_test_data(test_df, image_size=(256, 256)):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_test = []

    # Read and resize test images.
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']),
                                 total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size)
        x_test.append(img)

    x_test = np.array(x_test)

    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))

    return x_test


# Collection of methods for basic data manipulation like normalizing,
# inverting, color transformation and generating new images/masks

def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)


def normalize_masks(data):
    """Normalize masks."""
    return normalize(data, type_=1)


def normalize(data, type_=1):
    """Normalize data."""
    if type_ == 0:
        # Convert pixel values from [0:255] to [0:1] by global factor
        data = data.astype(np.float32) / data.max()
    if type_ == 1:
        # Convert pixel values from [0:255] to [0:1] by local factor
        div = data.max(axis=tuple(
            np.arange(1, len(data.shape))), keepdims=True)
        div[div < 0.01 * data.mean()] = 1.  # protect against too small pixel intensities
        data = data.astype(np.float32) / div
    if type_ == 2:
        # Standardisation of each image
        data = data.astype(np.float32) / data.max()
        mean = data.mean(axis=tuple(
            np.arange(1, len(data.shape))), keepdims=True)
        std = data.std(axis=tuple(
            np.arange(1, len(data.shape))), keepdims=True)
        data = (data - mean) / std

    return data


def trsf_proba_to_binary(y_data, threshold=0.5):
    """Transform propabilities into binary values 0 or 1."""
    return np.greater(y_data, threshold).astype(np.uint8)


def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(
        list(map(lambda x: 1. - x if np.mean(x) > cutoff else x, imgs)))
    return normalize_imgs(imgs)


def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs


def generate_images(imgs, seed=None):
    """Generate new images."""
    # Transformations.
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90., width_shift_range=0.02, height_shift_range=0.02,
        zoom_range=0.10, horizontal_flip=True, vertical_flip=True)

    # Generate new set of images
    imgs = image_generator.flow(imgs, np.zeros(len(imgs)),
                                batch_size=len(imgs),
                                shuffle=False, seed=seed).next()
    return imgs[0]


def generate_images_and_masks(imgs, masks, weights):
    """Generate new images and masks."""
    seed = np.random.randint(10000)
    imgs = generate_images(imgs, seed=seed)
    weights = generate_images(weights, seed=seed)
    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
    return imgs, masks, weights


def preprocess_raw_data(x_train, y_train, y_weights,
                        grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    y_train = trsf_proba_to_binary(normalize_masks(y_train))
    y_weights = normalize(y_weights, type_=0)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        print('Images inverted to remove light backgrounds.')

    return x_train, y_train, y_weights


def preprocess_test_data(x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_test = normalize_imgs(x_test)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return x_test


def load_images_from_path(paths, tgt_size=None, color=True):
    '''Load images from a list of paths'''
    images = []
    for path in paths:
        im = cv2.imread(
            path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if tgt_size is not None:
            im = cv2.resize(im, tgt_size)
        images.append(im)
    return images
