from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger

from .metrics_logger import MetricsLogger
from .x_logger import XLogger


def create_callbacks(
    val_ds,
    test_ds=None,
    val_freq=10,
    early_stopping=200,
    top_k=[10, 20],
    monitor='val_ndcg@20',
    ckpt_path=None,
    log_dir=None,
    x_sample=None,
    **kwargs,
):
    callbacks = []

    if 'loss' in monitor:
        monitor_mode = 'min'
    elif 'ndcg' in monitor:
        monitor_mode = 'max'
    elif 'recall' in monitor:
        monitor_mode = 'max'
    else:
        raise ValueError(f'Invalid metric: {monitor}')

    callbacks.append(MetricsLogger(val_ds, val_freq, top_k, prefix='val_'))
    if test_ds is not None:
        callbacks.append(MetricsLogger(test_ds, val_freq, top_k, prefix='test_'))

    if ckpt_path is not None:
        callbacks.append(
            ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True,
                monitor=monitor,
                mode=monitor_mode,
                save_best_only=True,
            )
        )

    if 'loss' in monitor:
        patience = early_stopping
    else:
        patience = early_stopping // val_freq
    callbacks.append(
        EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            monitor=monitor,
            mode=monitor_mode,
        )
    )

    if log_dir is not None and x_sample is not None:
        callbacks.append(XLogger(x=x_sample, log_dir=log_dir))

    if log_dir is not None:
        callbacks.append(TensorBoard(log_dir))
        callbacks.append(CSVLogger(f'{log_dir}.csv'))

    return callbacks
