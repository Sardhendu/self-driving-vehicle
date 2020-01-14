
import numpy as np
import matplotlib.pyplot as plt


def subplots(nrows, ncols, figsize=(50, 10)):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    if ncols > 1:
        ax = ax.ravel()
    else:
        ax = [ax]
    
    def _subplots(list_of_images, list_of_titles):
        if list_of_titles is not None:
            assert (len(list_of_images) == len(list_of_titles))
        else:
            list_of_titles = [f"img_{i}" for i in np.arange(list_of_images)]
            
        for i, (img, img_name) in enumerate(zip(list_of_images, list_of_titles)):
            ax[i].imshow(img)
            ax[i].title.set_text(img_name)
    return _subplots
