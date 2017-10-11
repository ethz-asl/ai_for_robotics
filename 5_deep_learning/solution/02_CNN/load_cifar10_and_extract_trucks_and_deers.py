####################################
# Author: Renaud Dube              #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Renaud Dube          #
####################################

import os
import pickle
import sys
import tarfile
import numpy as np
from six.moves import urllib

from IPython import embed

DEST_DIRECTORY = "/tmp/cifar10_data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DEST_DIRECTORY):
        os.makedirs(DEST_DIRECTORY)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DEST_DIRECTORY, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(DEST_DIRECTORY, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(DEST_DIRECTORY)


########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_width = 32
img_height = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_width * img_height * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(DEST_DIRECTORY, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file)

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_width, img_height])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.

    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(cls)-1.

    :return:
        2-dim array of shape: [len(cls), num_classes]
    """

    # Find the number of classes if None is provided.
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(
        shape=[_num_images_train, img_width, img_height, num_channels],
        dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(
            filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(
        class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(
        class_numbers=cls, num_classes=num_classes)


import matplotlib.pyplot as plt


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :], interpolation=interpolation)

        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name,
                                                   cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


########################################################################

maybe_download_and_extract()
images_train, cls_train, _ = load_training_data()
images_test, cls_test, _ = load_test_data()

class_names = load_class_names()
print(class_names)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
#print("- Test-set:\t\t{}".format(len(images_test)))

# Get the first images from the test-set.
images = images_train[0:9]

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Extract deers and trucks
indices_deers_and_trucks = np.where(
    np.logical_or(cls_train == 4, cls_train == 9))
images_train = images_train[indices_deers_and_trucks]
cls_train = cls_train[indices_deers_and_trucks]
print(cls_train[0:9])
cls_train[np.where(cls_train == 4)] = np.zeros(
    np.shape(np.where(cls_train == 4))[0])
cls_train[np.where(cls_train == 9)] = np.ones(
    np.shape(np.where(cls_train == 9))[0])
print(cls_train[0:9])

indices_deers_and_trucks = np.where(
    np.logical_or(cls_test == 4, cls_test == 9))
images_test = images_test[indices_deers_and_trucks]
cls_test = cls_test[indices_deers_and_trucks]
print(cls_test[0:9])
cls_test[np.where(cls_test == 4)] = np.zeros(
    np.shape(np.where(cls_test == 4))[0])
cls_test[np.where(cls_test == 9)] = np.ones(
    np.shape(np.where(cls_test == 9))[0])
print(cls_test[0:9])

# Remix the training and testing set so that students do not know what is our testing set.
images = np.concatenate(
    (images_train[0:8000, :, :, :], images_test, images_train[8000:, :, :, :]),
    axis=0)
cls = np.concatenate((cls_train[0:8000], cls_test, cls_train[8000:]), axis=0)

n_train = 7000
n_test = 2000
n_test_private = 3000

images_train = images[0:n_train, :, :, :]
cls_train = cls[0:n_train]
images_test = images[n_train:n_train + n_test, :, :, :]
cls_test = cls[n_train:n_train + n_test]
images_test_private = images[n_train + n_test:
                             n_train + n_test + n_test_private, :, :, :]
cls_test_private = cls[n_train + n_test:n_train + n_test + n_test_private]

print("Training set size", np.shape(images_train))
print("Testing set size", np.shape(images_test))
print("Private testing set size", np.shape(images_test_private))

# Get the first images from the test-set.
images = images_train[0:9]

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true, smooth=False)

#print(np.shape(images_train[0]))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def rgbs2grays(rgbds):
    grays = []
    for rgb in rgbs:
        grays.append(rgb2gray(rgb))
    return grays


images_train = rgb2gray(images_train)
images_test = rgb2gray(images_test)
images_test_private = rgb2gray(images_test_private)

data_train = {"images": images_train, "cls": cls_train}
data_test = {"images": images_test, "cls": cls_test}
data_test_private = {"images": images_test_private, "cls": cls_test_private}

if not os.path.exists("data/"):
    os.makedirs("data/")

with open(r"data/deers_and_trucks", "wb") as output_file:
    pickle.dump(data_train, output_file)

with open(r"data/deers_and_trucks_test", "wb") as output_file:
    pickle.dump(data_test, output_file)

with open(r"/tmp/deers_and_trucks_evaluation", "wb") as output_file:
    pickle.dump(data_test_private, output_file)
