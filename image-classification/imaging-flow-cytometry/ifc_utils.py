import glob
import math
import os
import os.path
import random
import warnings

import bioformats
import bioformats.formatreader
import javabridge
import keras.utils.np_utils
import numpy
import skimage.exposure
import skimage.io
import skimage.measure
import skimage.morphology


def class_weights(directory, data):
    """
    Compute the contribution of data from each class.

    :param directory: A directory containing class-labeled subdirectories containing .PNG images.
    :param data: A dictionary of class labels to directories containing .CIF files of that class. E.g.,
                     directory = {
                         "abnormal": "data/raw/abnormal",
                         "normal": "data/raw/normal"
                     }
    :return: A dictionary of class labels and contributions (as a decimal percentage), compatible with Keras.
    """
    counts = {}

    for label_index, label in enumerate(sorted(data.keys())):
        count = len(glob.glob("{}/{}/*.png".format(directory, label)))

        counts[label_index] = count

    total = max(sum(counts.values()), 1)

    for label_index, count in counts.items():
        counts[label_index] = count / total

    return counts


def parse(directory, data, channels):
    """
    Extracts single-channel .PNG images from .CIF files.

    Extracted images are saved to the following directory structure:
        directory/
            class_label_0/
                class_label_0_XX_YYYY_ZZ.png
                class_label_0_XX_YYYY_ZZ.png
                ...
            class_label_1/
                class_label_1_XX_YYYY_ZZ.png
                class_label_1_XX_YYYY_ZZ.png
                ...

    This directory structure can be processed by split to create training/validation and test sets.

    :param directory: The directory where extracted images are saved. The directory is assumed to be empty and will be
                      created if it does not exist.
    :param data: A dictionary of class labels to directories containing .CIF files of that class. E.g.,
                     directory = {
                         "abnormal": "data/raw/abnormal",
                         "normal": "data/raw/normal"
                     }
    :param channels: An array of channel indices (0 indexed). Only these channels are extracted. Unlisted channels are
                     ignored.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    javabridge.start_vm(class_path=bioformats.JARS)

    warnings.filterwarnings("ignore")

    for label, data_directory in data.items():
        if not os.path.exists("{}/{}".format(directory, label)):
            os.makedirs("{}/{}".format(directory, label))

        filenames = glob.glob("{}/*.cif".format(data_directory))

        for file_id, filename in enumerate(filenames):
            _parse_cif(filename, label, file_id, directory, channels)

    warnings.resetwarnings()

    # JVM can't be restarted once stopped.
    # Restart the notebook to run again.
    javabridge.kill_vm()


def split(directory, labels, split):
    """
    Shuffle and split image data into training/validation and test sets.

    Generates four files for use with training:
        directory/test_x.npy
        directory/test_y.npy
        directory/training_x.npy
        directory/training_y.npy

    :param directory: A directory containing class-labeled subdirectories containing single-channel .PNG or .TIF images.
    :param data: A list of class labels.
    :param split: Percentage of data (as a decimal) assigned to the training/validation set.
    """
    filenames = []

    labels = sorted(labels)

    for label in labels:
        label_pngs = glob.glob(os.path.join(directory, label, "*.png"))

        label_tifs = glob.glob(os.path.join(directory, label, "*.tif"))

        filenames = numpy.concatenate((filenames, label_pngs, label_tifs))

    random.shuffle(filenames)

    n_training = math.ceil(len(filenames) * split)

    training_filenames = filenames[:n_training]

    test_filenames = filenames[n_training:]

    for name, filenames in [("training", training_filenames), ("test", test_filenames)]:
        x, y = _concatenate(filenames, labels)

        numpy.save(os.path.join(directory, "{}_x.npy".format(name)), x)

        numpy.save(os.path.join(directory, "{}_y.npy".format(name)), y)


def _concatenate(filenames, labels):
    collection = skimage.io.imread_collection(filenames)

    x = collection.concatenate().reshape((-1, 32, 32, 1))

    y = [labels.index(os.path.split(os.path.dirname(filename))[-1]) for filename in filenames]

    return x, keras.utils.np_utils.to_categorical(y)


def _crop(image, mask):
    if min(image.shape) < 32:
        return None

    mask = mask > 0

    mask = skimage.morphology.remove_small_objects(mask, 4)

    if numpy.all(mask == 0):
        return None

    regionprops = skimage.measure.regionprops(skimage.measure.label(mask))

    bbox = regionprops[0].bbox

    if bbox[2] - bbox[0] > 32 or bbox[3] - bbox[1] > 32:
        return None

    center_x = int(image.shape[0] / 2.0)

    center_y = int(image.shape[1] / 2.0)

    cropped = image[center_x - 16:center_x + 16, center_y - 16:center_y + 16]

    assert cropped.shape == (32, 32), cropped.shape

    return cropped


def _parse_cif(filename, label, file_id, directory, channels):
    reader = bioformats.formatreader.get_image_reader("tmp", path=filename)

    image_count = javabridge.call(reader.metadata, "getImageCount", "()I")

    for index in range(image_count)[::2]:
        image = reader.read(series=index)

        mask = reader.read(series=index + 1)

        for channel in channels:
            cropped = _crop(image[:, :, channel], mask[:, :, channel])

            if cropped is None:
                continue

            rescaled = skimage.exposure.rescale_intensity(
                cropped,
                out_range=numpy.uint8
            ).astype(numpy.uint8)

            skimage.io.imsave(
                "{}/{}/{}_{:02d}_{:04d}_{:02d}.png".format(
                    directory,
                    label,
                    label,
                    file_id,
                    int(index / 2.0), channel
                ),
                rescaled,
                plugin="imageio"
            )
