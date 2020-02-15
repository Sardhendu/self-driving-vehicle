import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Callable
from src import commons
from src.traffic_sign_classifier.preprocess.augment import preprocess
from src.traffic_sign_classifier.model import LeNet
from src.traffic_sign_classifier import ops
from src.traffic_sign_classifier.params import params

label_to_class_mapping = pd.read_csv("./signnames.csv", header="infer")
label_to_class_mapping = label_to_class_mapping.to_dict()["SignName"]


def predict(
        image_dir,
        input_fn: Callable,
        model_builder: Callable,
        params: Dict
):
    optimizer_ = tf.keras.optimizers.Adam(0.1)
    checkpoints = ops.CheckpointCallback(model_dir=params["model_dir"], optimizer=optimizer_, model=model_builder)
    checkpoints.restore()
    
    store_plots = []
    img_names = []
    for path_ in [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i.endswith("jpeg")]:
        print('\nRunning for path: ......', path_)
        features = commons.read_image(path_)
        features = cv2.resize(features, (params["img_height"], params["img_width"]))
        store_plots.append(np.uint8(features))
        img_names.append(os.path.basename(path_).split(".")[0])
        
        features_pp = tf.constant(features, dtype=tf.float32)
        features_pp, _ = input_fn(features_pp, None, params["num_classes"])
        
        features_pp = tf.expand_dims(features_pp, axis=0)
        
        pred_logits = model_builder(features_pp)
        pred_prob = tf.nn.softmax(pred_logits)
        pred_prob = tf.squeeze(pred_prob, axis=0)
        pred_score, pred_classes = tf.nn.top_k(pred_prob, 5)
        print('Pred Labels: ', pred_classes.numpy())
        print('Pred Classes ', [label_to_class_mapping[i] for i in pred_classes.numpy()])
        print('Pred Scores: ', np.round(pred_score.numpy(), 3))
        
    fig = commons.image_subplots(nrows=1, ncols=5, figsize=(3, 6))(store_plots, img_names)
    commons.save_matplotlib("./images/test_images.png", fig)
    
    
if __name__ == "__main__":
    test_data_path = "./data/test.p"
    test_data = commons.read_pickle(test_data_path)
    
    test_features = test_data["features"]
    test_labels = test_data["labels"]
    print(f"[Train]: features={test_features.shape}, labels={test_labels.shape}")
    
    predict_preprocess = preprocess(mode="predict")
    # test_dataset_ = dataset_pipeline(test_features, test_labels, test_preprocess, params, mode="test")
    model_fn_ = LeNet(num_classes=43)
    predict(image_dir="./predict_images", input_fn=predict_preprocess, model_builder=model_fn_, params=params)
