import os
import cv2
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt

from typing import Tuple


class ImagePlots:
    def __init__(self, image):
        self.orig_img = image.copy()
        self.image = image.copy()
        
    def polylines(self, points: np.array, color: Tuple[int, int, int] = (50, 255, 255)):
        assert (points.shape[1] == 2)
        cv2.polylines(self.image, [points], False, color, thickness=4)
        
    def polymask(self, points: np.array, color: Tuple[int] = (50, 255, 255), mask_weight: float = 0.5):
        assert (points.shape[1] == 2)
        cv2.fillPoly(self.image, [points], color)
        cv2.addWeighted(self.orig_img, mask_weight, self.image, 1 - mask_weight, 0, self.image)
    
        
def subplots(nrows=1, ncols=1, figsize=(6, 6), fontsize=25, facecolor='w'):
    figsize = tuple([max(figsize[0], ncols*6), max(figsize[1], nrows*4)])
    print(figsize)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=facecolor)
    
    if ncols > 1:
        ax = ax.ravel()
    else:
        ax = [ax]
    
    def _subplots(list_of_images, list_of_titles=None):
        if list_of_titles is not None:
            assert (len(list_of_images) == len(list_of_titles))
        else:
            list_of_titles = [f"img_{i}" for i in np.arange(len(list_of_images))]
            
        for i, (img, img_name) in enumerate(zip(list_of_images, list_of_titles)):
            ax[i].imshow(img)
            ax[i].set_title(img_name,  fontsize=fontsize)
        return fig
    return _subplots


def basic_plot(nrows=1, ncols=1, figsize=(6, 6), fontsize=25, facecolor='w'):
    figsize = tuple([max(figsize[0], ncols * 6), max(figsize[1], nrows * 4)])
    print(figsize)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=facecolor)
    
    if ncols > 1:
        ax = ax.ravel()
    else:
        ax = [ax]
    
    def _subplots(list_of_vectors, list_of_titles=None):
        if list_of_titles is not None:
            assert (len(list_of_vectors) == len(list_of_titles))
        else:
            list_of_titles = [f"graph_{i}" for i in np.arange(len(list_of_vectors))]
        
        for i, (vec, img_name) in enumerate(zip(list_of_vectors, list_of_titles)):
            ax[i].plot(vec)
            ax[i].set_title(img_name, fontsize=fontsize)
        return fig
    
    return _subplots


def save_matplotlib(image_path, fig):
    image_dir = os.path.dirname(image_path)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    if image_path.endswith("jpg"):
        image_path = image_path[0:-3]+".png"
    fig.savefig(image_path)
    

def save_image(image_path, image):
    image_dir = os.path.dirname(image_path)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    imageio.imsave(image_path, np.array(image).astype(np.uint8))
    
    
def read_image(image_path):
    return imageio.imread(image_path)

    
def read_pickle(save_path):
    with open(save_path, "rb") as p:
        data = pickle.load(p)
    return data


def write_pickle(save_path, data_dict: dict):
    folder_dir = os.path.dirname(save_path)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)

