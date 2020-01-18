import os
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt


def subplots(nrows, ncols, figsize=(50, 10), fontsize=25, facecolor='w'):
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

