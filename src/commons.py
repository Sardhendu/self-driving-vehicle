import os
import cv2
import pickle
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List
from moviepy.editor import VideoFileClip


class ImagePlots:
    def __init__(self, image):
        self.orig_img = image.copy()
        self.image = image.copy()
        
    def polylines(self, points: np.array, color: Tuple[int, int, int] = (50, 255, 255), thickness: int = 4):
        assert (points.shape[1] == 2)
        cv2.polylines(self.image, [points], False, color, thickness=thickness)
        
    def polymask(self, points: np.array, color: Tuple[int] = (50, 255, 255), mask_weight: float = 0.5):
        assert (points.shape[1] == 2)
        cv2.fillPoly(self.image, [points], color)
        cv2.addWeighted(self.orig_img, mask_weight, self.image, 1 - mask_weight, 0, self.image)
        
    def rectangle(self, bbox: List, color: Tuple = None):
        color = (255, 0, 0) if color is None else color
        assert(len(bbox) == 4)
        cv2.rectangle(self.image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
        
    def point(self, point: List[int], color: Tuple = None):
        color = (0, 0, 255) if color is None else color
        cv2.circle(self.image, tuple(point), 3, color, 3)
        
    def add_caption(self, caption: str, pos: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2):
        cv2.putText(self.image, caption, pos, cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=thickness)
        

def fetch_image_from_video(input_video_path, output_img_dir, time_list=[0.24, 0.243, 0.245, 0.248, 0.25]):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    
    clip = VideoFileClip(input_video_path)
    for t in time_list:
        imgpath = os.path.join(output_img_dir, '{}.jpg'.format(t))
        clip.save_frame(imgpath, t)


def image_subplots(nrows=1, ncols=1, figsize=(6, 6), fontsize=25, facecolor='w'):
    figsize = tuple([max(figsize[0], ncols*6), max(figsize[1], nrows*4)])
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


def graph_subplots(nrows=1, ncols=1, figsize=(6, 6), fontsize=25, ticks_fontsize=15, facecolor='w'):
    figsize = tuple([max(figsize[0], ncols * 6), max(figsize[1], nrows * 4)])
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
            for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
                         ax[i].get_xticklabels() + ax[i].get_yticklabels()):
                item.set_fontsize(ticks_fontsize)
        return fig
    
    return _subplots


def pandas_subplots(nrows=1, ncols=1, figsize=(6, 6), fontsize=25, facecolor='w'):
    figsize = tuple([max(figsize[0], ncols * 6), max(figsize[1], nrows * 4)])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=facecolor)

    if ncols > 1:
        ax = ax.ravel()
    else:
        ax = [ax]

    def _subplots(list_of_df, list_of_titles=None):
        if list_of_titles is not None:
            assert (len(list_of_df) == len(list_of_titles))
        else:
            list_of_titles = [f"graph_{i}" for i in np.arange(len(list_of_df))]

        for i, (df, img_name) in enumerate(zip(list_of_df, list_of_titles)):
            df.plot(ax=ax[i])
            ax[i].set_title(img_name, fontsize=fontsize)
        return fig

    return _subplots


def plot_confusion_matrix(x, color_map='YlGnBu'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    sns.heatmap(x, cmap=color_map, annot=True, ax=ax, annot_kws={"fontsize":5})
    ax.set_title("confusion_matrix", fontsize=20)

    for item in (
            [ax.title, ax.xaxis.label,
             ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(5)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.show()


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


def write_gif(video_path, subclip_from, subclip_to, resize, speedx, save_path):
    clip = (VideoFileClip(video_path)
            .subclip(subclip_from, subclip_to)
            .speedx(speedx)
            .resize(resize))
    clip.write_gif(save_path)
    
    
if __name__ == "__main__":
    out_dir = "/Users/sardhendu/workspace/self-driving-vehicle/src/extended_kalman_filters/images"
    write_gif(
            video_path="/Users/sardhendu/Desktop/ekf.mov",
            subclip_from=0,
            subclip_to=12,
            resize=0.5,
            speedx=4.0,
            save_path=f"{out_dir}/sneak_peak.gif")
