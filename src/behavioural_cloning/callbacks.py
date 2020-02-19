import tensorflow as tf
import os

from src.traffic_sign_classifier.ops import SummaryCallback


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, params, verbose=0):
        super().__init__()
        self.verbose = verbose
        self.log_at_step = params["train_log_steps"]
        self.file_writer = tf.summary.create_file_writer(params['tensorboard_dir'] + "/train")
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if (self.step % self.log_at_step) == 0 or (self.step == 1):
            self.file_writer.set_as_default()
            for name, value in logs.items():
                if "loss" in name:
                    tf.summary.scalar("loss", data=value.item(), step=self.step)
                    if self.verbose > 0:
                        print('\nBatch=%s, Step=%s: Training Callback setting %s '
                              'to %s.' % (str(batch), str(self.step), str(name), str(value.item())))

    # def on_epoch_end(self, epoch, logs=None):
    #     pass


class SnapshotCallback(tf.keras.callbacks.Callback):
    def __init__(self, params, verbose=1):
        """
        self.model.save             : will save the full model (including the graph)
        self.model.save_weights     : will save only weights
        """
        super().__init__()
        self.backbone = params["backbone"]
        self.log_at_step = params["snapshot_log_steps"]
        self.snapshot_dir = params["snapshot_dir"]
        self.verbose = verbose
        self.step = 0
        
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if (self.step % self.log_at_step) == 0:
            name = '%s_train_step_%s.h5' % (str(self.backbone), str(self.step))

            if self.verbose == 1:
                print('[Snapshot] Model checkpoints at: ', os.path.join(self.snapshot_dir, name))
            self.model.save(os.path.join(self.snapshot_dir, name))
    
    # def on_epo s


class TensorBoardCallback:
    def __init__(self, params):
        self.params = params
        
    def __call__(self):
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.params['tensorboard_dir'],
            histogram_freq=0,
            batch_size=self.params['train_batch_size'],
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        return tb_callback


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, eval_dataset, params):
        self.params = params
        self.log_at_step = params["eval_log_steps"]
        self.eval_dataset = eval_dataset
        self.model = model
        self.step = 0
        self.flatten = tf.keras.layers.Flatten()
        self.loss_writer = SummaryCallback(params["tensorboard_dir"] + "/eval")

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        print("self.log_at_step: ", self.log_at_step)
        if (self.step % self.log_at_step) == 0:
            metric = tf.keras.metrics.Mean()
            print(len(self.eval_dataset))
            print(1/0)
            # tf.keras.utils.Progbar()
            for image_batch, label_batch in self.eval_dataset:
                pred_batch = self.model(image_batch)
                mse_loss = tf.keras.losses.MSE(label_batch, self.flatten(pred_batch))
                val_batch_loss = tf.reduce_mean(mse_loss)
                metric.update_state(val_batch_loss)
                
            self.loss_writer.scalar("loss", metric.result(), self.step )
            