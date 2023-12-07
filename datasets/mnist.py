import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

from .utils import unpickle


def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        # read the magic number and the number of images
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        num_images = int.from_bytes(file.read(4), byteorder='big')
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_columns = int.from_bytes(file.read(4), byteorder='big')

        # read the image data
        image_data = np.fromfile(file, dtype=np.uint8)
        image_data = image_data.reshape(num_images, num_rows, num_columns)

    return image_data

def read_mnist_labels(filename):
    with open(filename, 'rb') as file:
        # read the magic number and the number of labels
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        num_labels = int.from_bytes(file.read(4), byteorder='big')

        # read the label data
        label_data = np.fromfile(file, dtype=np.uint8)

    return label_data


def load_mnist(root):

    train_images_file = os.path.join(root, 'train-images.idx3-ubyte')
    train_labels_file = os.path.join(root, 'train-labels.idx1-ubyte')
    test_images_file = os.path.join(root, 't10k-images.idx3-ubyte')
    test_labels_file = os.path.join(root, 't10k-labels.idx1-ubyte')
    x_train = read_mnist_images(train_images_file)
    y_train = read_mnist_labels(train_labels_file)
    x_test = read_mnist_images(test_images_file)
    y_test = read_mnist_labels(test_labels_file)
    
    return x_train, y_train, x_test, y_test





class MNIST(Dataset):
    
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.images, self.labels = self.get_files()
        assert len(self.images) == len(self.labels)

    def read_images(self, filename):
        with open(filename, 'rb') as file:
            # read the magic number and the number of images
            magic_number = int.from_bytes(file.read(4), byteorder='big')
            num_images = int.from_bytes(file.read(4), byteorder='big')
            num_rows = int.from_bytes(file.read(4), byteorder='big')
            num_columns = int.from_bytes(file.read(4), byteorder='big')

            # read the image data
            image_data = np.fromfile(file, dtype=np.uint8)
            image_data = image_data.reshape(num_images, num_rows * num_columns)
        
        return image_data


    def read_labels(self, filename):
        with open(filename, 'rb') as file:
            # read the magic number and the number of labels
            magic_number = int.from_bytes(file.read(4), byteorder='big')
            num_labels = int.from_bytes(file.read(4), byteorder='big')

            # read the label data
            label_data = np.fromfile(file, dtype=np.uint8)
        
        return label_data

        
    def get_files(self):

        if self.train:
            images = self.read_images(os.path.join(self.root, 'train-images.idx3-ubyte'))
            labels = self.read_labels(os.path.join(self.root, 'train-labels.idx1-ubyte'))
        else:
            images = self.read_images(os.path.join(self.root, 't10k-images.idx3-ubyte'))
            labels = self.read_labels(os.path.join(self.root, 't10k-labels.idx1-ubyte'))

        return images, labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
    
    def get_images(self):
        return self.images
    
    def get_labels(self):
        return self.labels
    
    def show_sample_images(self, num_samples=5):
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
        for i in range(num_samples):
            img = self.images[i].reshape(28, 28)
            axs[i].imshow(img)
            axs[i].set_title(self.labels[i])
            axs[i].axis('off')
        plt.show()

