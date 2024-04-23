from models import Recommender
from keras.callbacks import Callback


class MetricsLogger(Callback):
    '''
    Report NDCG@k or Recall@k on the validation set.
    '''

    def __init__(self, ds, freq=1, top_k=[], prefix=''):
        super().__init__()
        self.model: Recommender
        self.ds = ds
        self.freq = freq
        self.top_k = top_k
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        results = self.model.evaluate(self.ds, self.top_k, return_dict=True)
        for metric, value in results.items():
            metric = f'{self.prefix}{metric}'
            logs[metric] = value
