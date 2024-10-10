import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.ndimage import rotate as scipy_rotate
import random

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augmentation_factor=1):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augmentation_factor = augmentation_factor

    def __len__(self):
        return len(self.images) * self.augmentation_factor

    def __getitem__(self, idx):
        original_idx = idx % len(self.images)
        image = self.images[original_idx]
        label = self.labels[original_idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def rotate_with_reflect(img, angle):
    # Get dimensions of the original image
    h, w = img.shape[:2]

    # Reflect padding
    pad_h = int((np.sqrt(2) * max(h, w) - h) / 2)
    pad_w = int((np.sqrt(2) * max(h, w) - w) / 2)

    img_reflect = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    
    # Rotate the image by arbitrary angle
    img_rotated = scipy_rotate(img_reflect, angle, reshape=False, mode='reflect')
    
    # Crop to the original size
    left = pad_w
    top = pad_h
    right = pad_w + w
    bottom = pad_h + h
    
    img_cropped = img_rotated[top:bottom, left:right]
    return img_cropped


class RebinTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def rebin_channel(self, a, shape):
        '''
        Re-bin a single channel array into a new shape, and take the average
        '''
        sh = (shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1])
        return a.reshape(sh).mean(-1).mean(1)

    def __call__(self, img):

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")


        rebinned_array = np.zeros((self.new_shape[0], self.new_shape[1], 3), dtype=np.float32)

        for c in range(3):
            rebinned_array[:, :, c] = self.rebin_channel(img[:, :, c], self.new_shape)
        
        return rebinned_array

def horizontal_flip(img):
    return np.flip(img, axis=1)

def vertical_flip(img):
    return np.flip(img, axis=0)


new_shape = (20, 20)


rebin_transform = RebinTransform(new_shape)

transform_resize = transforms.Compose([
    # transforms.Lambda(lambda img: rebin_transform(img)),  # you can rebin by RebinTransform or KANconv, KAnconv perform better
    transforms.Lambda(lambda img: img.astype(np.float32)), 
    transforms.Lambda(lambda img: horizontal_flip(img) if random.random() > 0.5 else img),
    transforms.Lambda(lambda img: vertical_flip(img) if random.random() > 0.5 else img),
    transforms.Lambda(lambda img: rotate_with_reflect(img, angle=random.uniform(0, 360))),
    transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)).copy()))
])

def process_data(images, labels, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=random_state, stratify=labels)
    unique, counts = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)

    return X_train, X_val, y_train, y_val


def getGenerators(X_train, X_val, y_train, y_val):
    train_generator = CustomDataset(X_train, y_train, transform=transform_resize)
    val_generator = CustomDataset(X_val, y_val, transform=transform_resize)
    train_loader = DataLoader(train_generator, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_generator, batch_size=32, shuffle=False)
    return train_loader, val_loader



def get_predictions_per_subset(probability, n_samples_per_subset, cross_sections=[0., 0.1, 1.0], return_weights=False):
    cross_sections = np.array(cross_sections)
    nMonte_carlo = probability.shape[0]
    nClusters = probability.shape[1]
    nDM_Models = probability.shape[2]

    nSubSets = nClusters // n_samples_per_subset
    subset_means, subset_stds, prediction, prediction_err = [], [], [], []

    for iSubSet in range(nSubSets):
        this_subset = np.arange(iSubSet * n_samples_per_subset, min([(iSubSet + 1) * n_samples_per_subset, nClusters]))
        final_probs = np.ones((nMonte_carlo, nDM_Models))

        for iMonteCarlo in range(nMonte_carlo):
            final_probs_per_cluster_per_MC = probability[iMonteCarlo, this_subset, :]
            final_probs_per_cluster_per_MC = final_probs_per_cluster_per_MC / np.sum(final_probs_per_cluster_per_MC, axis=1)[:, np.newaxis]

            for iCluster in range(final_probs_per_cluster_per_MC.shape[0]):
                final_probs[iMonteCarlo] *= final_probs_per_cluster_per_MC[iCluster]
                final_probs[iMonteCarlo] /= np.sum(final_probs[iMonteCarlo])

        final_probs_all = np.mean(final_probs, axis=0)

        if return_weights:
            prediction.append(final_probs)
        else:
            pred = np.average(cross_sections, weights=final_probs_all)
            prediction.append(pred)
            prediction_err.append(np.sqrt(np.sum(final_probs_all * (cross_sections - pred) ** 2) / np.sum(final_probs_all) / n_samples_per_subset))

    return np.array(prediction), np.array(prediction_err)