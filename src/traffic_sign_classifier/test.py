import time
import numpy as np
import tensorflow as tf
from typing import Callable, Dict
from sklearn import metrics

from src import commons
from src.traffic_sign_classifier import ops
from src.traffic_sign_classifier.params import params
from src.traffic_sign_classifier.model import LeNet, dataset_pipeline
from src.traffic_sign_classifier.preprocess.augment import preprocess


def confusion_matrix(n_classes):
    def _confusion_matrix(y_true, y_pred):
        conf_mat = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(n_classes), normalize="true")
        return np.round(conf_mat, 2)
    return _confusion_matrix

    
def test(
        dataset_: Callable,
        model_builder: Callable,
        params: Dict
):
    optimizer_ = tf.keras.optimizers.Adam(0.1)
    checkpoints = ops.CheckpointCallback(model_dir=params["model_dir"], optimizer=optimizer_, model=model_builder)
    accuracy = tf.keras.metrics.Accuracy(name="test_accuracy", dtype=tf.float32)
    auc = tf.keras.metrics.AUC(name="test_auc", dtype=tf.float32)
    pr = ops.PrecisionRecall(params["num_classes"], threshold=None)
    checkpoints.restore()
    dataset_iterator = iter(dataset_)
    
    all_labels = []
    all_preds = []
    for pp in range(0, 30):
        print('Running Batch = ', pp)
        feature, target = next(dataset_iterator)
        pred_logits = model_builder(feature)
        pred_prob = tf.nn.softmax(pred_logits)

        labels = tf.argmax(target, axis=-1)
        preds = tf.argmax(pred_prob, axis=-1)

        labels_idx = tf.stack([
            tf.cast(tf.range(tf.shape(feature)[0]), dtype=tf.int64), tf.cast(labels, dtype=tf.int64)
        ], axis=1)
        
        class_pred_prob = tf.gather_nd(pred_prob, labels_idx)
        accuracy.update_state(labels, preds)
        auc.update_state(labels, class_pred_prob)
        pr.update_state(labels, pred_prob)
        
        all_labels += list(labels.numpy())
        all_preds += list(preds.numpy())
    
    assert(len(all_labels) == len(all_preds))
    match = sum([1 for i, j in zip(all_labels, all_preds) if i == j])

    confusion_m = confusion_matrix(n_classes=43)(all_labels, all_preds)
    commons.plot_confusion_matrix(confusion_m)
    print("Accuracy: ", match/len(all_preds))
    print("\n Confusion Matrix \n: ", print(confusion_m))
    print(accuracy.result())
    print(auc.result())
    print(pr.result())
    

if __name__ == "__main__":
    test_data_path = "./data/test.p"
    test_data = commons.read_pickle(test_data_path)
    params["batch_size"] = 421
    test_features = test_data["features"]
    test_labels = test_data["labels"]
    print(f"[Train]: features={test_features.shape}, labels={test_labels.shape}")

    test_preprocess = preprocess(mode="eval")
    test_dataset_ = dataset_pipeline(test_features, test_labels, test_preprocess, params, mode="test")
    model_fn_ = LeNet(num_classes=43)
    test(test_dataset_, model_fn_, params)

