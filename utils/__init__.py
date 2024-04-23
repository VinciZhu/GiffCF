import os
import sys
import random
import numpy as np
import tensorflow as tf

from .data import *  # noqa: F403
from .config import *  # noqa: F403


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def init_tensorflow(verbose=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    physical_gpus = tf.config.list_physical_devices('GPU')
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    if verbose > 0:
        print(f'TensorFlow version: {tf.version.VERSION}, GPU: {logical_gpus}')
