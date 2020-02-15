import os
import cv2
import tensorflow as tf
from typing import Dict, Callable
from src import commons
from src.traffic_sign_classifier.preprocess.augment import preprocess
from src.traffic_sign_classifier.model import LeNet
from src.traffic_sign_classifier import ops
from src.traffic_sign_classifier.params import params


def predict(
        image_dir,
        input_fn: Callable,
        model_builder: Callable,
        params: Dict
):
    optimizer_ = tf.keras.optimizers.Adam(0.1)
    checkpoints = ops.CheckpointCallback(model_dir=params["model_dir"], optimizer=optimizer_, model=model_builder)
    checkpoints.restore()
    
    for path_ in [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i.endswith("jpeg")]:
        print('Running for path: ', path_)
        features = commons.read_image(path_)
        features = cv2.resize(features, (params["img_height"], params["img_width"]))
        
        features_pp = tf.constant(features, dtype=tf.float32)
        features_pp, _ = input_fn(features_pp, None, params["num_classes"])
        features_pp = tf.expand_dims(features_pp, axis=0)
        
        pred_logits = model_builder(features_pp)
        pred_prob = tf.nn.softmax(pred_logits)
        pred_prob = tf.squeeze(pred_prob, axis=0)
        # print("pred_prob: ", tf.argmax(pred_prob), tf.reduce_max(pred_prob))
        pred_score, pred_classes = tf.nn.top_k(pred_prob, 3)
        print(pred_classes.numpy())
        print(pred_score.numpy())
    #     print('Running Batch = ', pp)
    #     feature, target = next(dataset_iterator)
    #     pred_logits = model_builder(feature)
    #     pred_prob = tf.nn.softmax(pred_logits)
    #
    #     labels = tf.argmax(target, axis=-1)
    #     preds = tf.argmax(pred_prob, axis=-1)
    #
    #     labels_idx = tf.stack([
    #         tf.cast(tf.range(tf.shape(feature)[0]), dtype=tf.int64), tf.cast(labels, dtype=tf.int64)
    #     ], axis=1)
    #
    #     class_pred_prob = tf.gather_nd(pred_prob, labels_idx)
    #     accuracy.update_state(labels, preds)
    #     auc.update_state(labels, class_pred_prob)
    #     pr.update_state(labels, pred_prob)
    #
    #     all_labels += list(labels.numpy())
    #     all_preds += list(preds.numpy())
    #
    # assert (len(all_labels) == len(all_preds))
    # match = sum([1 for i, j in zip(all_labels, all_preds) if i == j])
    #
    # confusion_m = confusion_matrix(n_classes=43)(all_labels, all_preds)
    # commons.plot_confusion_matrix(confusion_m)
    # print("Accuracy: ", match / len(all_preds))
    # print("\n Confusion Matrix \n: ", print(confusion_m))
    # print(accuracy.result())
    # print(auc.result())
    # print(pr.result())


if __name__ == "__main__":
    test_data_path = "./data/test.p"
    test_data = commons.read_pickle(test_data_path)
    
    test_features = test_data["features"]
    test_labels = test_data["labels"]
    print(f"[Train]: features={test_features.shape}, labels={test_labels.shape}")
    
    predict_preprocess = preprocess(mode="predict")
    # test_dataset_ = dataset_pipeline(test_features, test_labels, test_preprocess, params, mode="test")
    model_fn_ = LeNet(num_classes=43)
    predict(image_dir="./data/predict_images", input_fn=predict_preprocess, model_builder=model_fn_, params=params)
