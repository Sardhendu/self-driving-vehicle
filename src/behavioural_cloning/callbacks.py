import os
import keras
import numpy as np
import tensorflow as tf
import math
from sklearn import metrics


class TensorBoardCallback:
    def __init__(self, params):
        self.params = params
    
    def __call__(self):
        tb_callback = keras.callbacks.TensorBoard(
                log_dir=self.params['tensorboard_dir'],
                histogram_freq=0,
                batch_size=self.params['batch_size'],
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None
        )
        return tb_callback


class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, tensorboard_callback, params, verbose=0):
        self.verbose = verbose
        self.log_at_step = params["train_log_steps"]
        self.tensorboard = tensorboard_callback
        
        super(TrainingCallback, self).__init__()
    
    def on_train_begin(self, logs=None):
        self.step = 0
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if (self.step % self.log_at_step) == 0 or (self.step == 1):
            
            if self.tensorboard is not None and self.tensorboard.writer is not None:
                for name, value in logs.items():
                    if name in ['batch', 'size']:
                        continue
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = '%s/%s' % (str('train'), str(name))
                    self.tensorboard.writer.add_summary(summary, self.step)
                    
                    if self.verbose > 0:
                        print('\nBatch=%s, Step=%s: Training Callback setting %s '
                              'to %s.' % (str(batch), str(self.step), str(name), str(value.item())))


class SnapshotCallback(keras.callbacks.Callback):
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
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if (self.step % self.log_at_step) == 0:
            name = '%s_train_step_%s.h5' % (str(self.backbone), str(self.step))
            
            if self.verbose == 1:
                print('[Snapshot] Model checkpoints at: ', os.path.join(self.snapshot_dir, name))
            self.model.save(os.path.join(self.snapshot_dir, name))


class ValidationCallback(keras.callbacks.Callback):
    def __init__(self, tensorboard_callback, model, eval_generator, params, verbose=0):
        self.params = params
        self.log_at_step = params["eval_log_steps"]
        self.eval_batch_cnt = params["eval_batch_cnt"]
        self.eval_generator = eval_generator
        self.model = model
        self.step = 0
        self.flatten = keras.layers.Flatten()
        self.tensorboard = tensorboard_callback
        self.verbose = verbose
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        if (self.step % self.log_at_step) == 0:
            total_eval_loss = 0
            
            eval_batch_cnt = min(self.eval_batch_cnt, len(self.eval_generator))
            prog = keras.utils.Progbar(eval_batch_cnt)
            
            for iter_ in range(0, eval_batch_cnt):
                image_batch, label_batch = self.eval_generator[iter_]
                pred_batch = self.model.predict_on_batch(image_batch)
                mse_loss = metrics.mean_squared_error(label_batch.flatten(), pred_batch.flatten())
                val_batch_loss = np.mean(mse_loss)
                total_eval_loss += val_batch_loss
                prog.update(iter_)
            
            total_eval_loss /= eval_batch_cnt
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = total_eval_loss
            summary_value.tag = '%s/%s' % (str('eval'), "loss")
            self.tensorboard.writer.add_summary(summary, self.step)
            
            if self.verbose > 0:
                print('\nBatch=%s, Step=%s: Training Callback setting %s '
                      'to %s.' % (str(batch), str(self.step), str("loss"), str(total_eval_loss.item())))


def poly_decay(initial_lrate=0.001, total_steps=5000, end_learning_rate=0, learning_power=0.9,
               learning_rate_min=0.000001):
    """
    :param initial_lrate:           The initial learning rate
    :param total_steps:             Total number of steps = epochs * steps_per_epoch
    :param end_learning_rate:
    :param learning_power:          Minimum learning power (Decides the polynomial decay) 0.9 is good for most cases
    :return:
    """
    
    def _poly_decay(step):
        global_step = min(step, total_steps)
        lrate = float((initial_lrate - end_learning_rate) * math.pow(1 - ((global_step) / total_steps),
                                                                     learning_power) + end_learning_rate)
        return max(lrate, learning_rate_min)
    
    return _poly_decay


class LearningRateSchedular(keras.callbacks.Callback):
    """
        This function is actually not required for training and testing, but is helpful, to get the underlying
        operation in of Keras loss and Learning Rate Decay
    """
    
    def __init__(self, decay_func, tensorboard_callback, params, verbose=0):
        # self.lr = []
        self.tensorboard = tensorboard_callback
        self.schedule = decay_func
        self.verbose = verbose
        self.log_at_step = params["train_log_steps"]
        super().__init__()
    
    def on_train_begin(self, logs=None):
        self.step = 0
    
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        
        try:  # new API
            lr = self.schedule(self.step, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(self.step)
        
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if (self.step % self.log_at_step) == 0 or (self.step == 1):
            if self.tensorboard is not None and self.tensorboard.writer is not None:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = lr  # self.mean_ap
                summary_value.tag = "learning_rate"
                
                self.tensorboard.writer.add_summary(summary, self.step)
            
            if self.verbose > 0:
                print('\nBatch=%s, Step=%s: LearningRateScheduler setting learning '
                      'rate to %s.' % (str(batch), str(self.step), lr))
