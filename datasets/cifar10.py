import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

from .utils import unpickle



class CIFAR10(Dataset):
    
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        dataset_split = 'train' if self.train else 'test'
        self.images, self.labels = self.get_files(dataset_split)
        assert len(self.images) == len(self.labels)


    def get_files(self, dataset_split):

        dataset_path = os.path.join(self.root, dataset_split)
        filenames = sorted(glob.glob(os.path.join(dataset_path,'*')))
        images = np.empty((0, 3072), dtype=np.uint8) 
        labels = np.empty((0), dtype=np.uint8)

        for batch_file in filenames:
            batch_data = unpickle(batch_file)
            batch_images = batch_data['data']
            batch_labels = batch_data['labels']

            images = np.concatenate((images, batch_images), axis=0)
            labels = np.concatenate((labels, batch_labels), axis=0)

        
        # to convert to 32x32x3, if desired. note this does mess up show_sample_images
        #images = images.reshape(images.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
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
            img = self.images[i].reshape(3, 32, 32).transpose(1, 2, 0)
            axs[i].imshow(img)
            axs[i].set_title(self.label_names[self.labels[i]])
            axs[i].axis('off')
        plt.show()

