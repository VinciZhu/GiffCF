import numpy as np
import tensorflow as tf
from models import Recommender
from keras.callbacks import Callback


class XLogger(Callback):
    '''
    Visualize (reconstructed) interaction vectors.
    '''

    def __init__(self, x, log_dir, freq=1):
        super().__init__()
        self.model: Recommender
        self.x = x
        self.width = self.height = int(np.sqrt(x.shape[1]))
        self.writer = tf.summary.create_file_writer(log_dir)
        self.freq = freq

    def on_train_begin(self, logs=None):
        image = self.x[:, : self.width * self.height].reshape(
            -1, self.width, self.height, 1
        )
        with self.writer.as_default():
            tf.summary.image('x', image, step=0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        x_pred = self.model.reconstruct(self.x).numpy()
        image = x_pred[:, : self.width * self.height].reshape(
            -1, self.width, self.height, 1
        )
        with self.writer.as_default():
            tf.summary.image('x_pred', image, step=epoch)

    def on_train_end(self, logs=None):
        self.writer.close()
