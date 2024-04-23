import numpy as np

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam


class Recommender(Model):
    # Multinomial cross entropy
    @staticmethod
    def cross_entropy(x, p, reduction='sum'):
        ce = -x * tf.math.log(p)
        if reduction == 'sum':
            return tf.reduce_mean(tf.reduce_sum(ce, axis=1))
        elif reduction == 'mean':
            return tf.reduce_mean(tf.reduce_mean(ce, axis=1))
        else:
            raise ValueError(f'Invalid reduction: {reduction}')

    # Mean squared error
    @staticmethod
    def squared_error(x, x_pred, reduction='sum'):
        se = tf.square(x - x_pred)
        if reduction == 'sum':
            return tf.reduce_mean(tf.reduce_sum(se, axis=1))
        elif reduction == 'mean':
            return tf.reduce_mean(tf.reduce_mean(se, axis=1))
        else:
            raise ValueError(f'Invalid reduction: {reduction}')

    # Return the reconstructed interaction vector.
    def reconstruct(self, x):
        return self(x, training=False)

    # Return the indices of top-k items.
    @tf.function
    def recommend(self, x, k, mask):
        scores = tf.where(mask, -np.inf, self(x, training=False))
        return tf.math.top_k(scores, k).indices

    def evaluate(
        self,
        ds: tf.data.Dataset,
        top_k=[],
        test_batch_size=400,
        return_dict=False,
        **kwargs,
    ):
        total_size = ds.cardinality().numpy()
        ds = ds.batch(test_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        max_k = max(top_k)
        rcpl_vals = 1 / np.log2(np.arange(2, max_k + 2))
        rcpl_cumsums = np.cumsum(rcpl_vals)
        rcpl_vals = tf.constant(rcpl_vals, dtype=tf.float32)
        rcpl_cumsums = tf.constant(rcpl_cumsums, dtype=tf.float32)
        rr_vals = tf.constant(1 / np.arange(1, max_k + 1), dtype=tf.float32)
        ndcg_sum = np.zeros(len(top_k))
        recall_sum = np.zeros(len(top_k))
        mrr_sum = np.zeros(len(top_k))
        for x_history, x_target, mask in ds:
            pred_items = self.recommend(x_history, max_k, mask)
            hit_flags = tf.gather(x_target, pred_items, axis=1, batch_dims=1)
            target_sum = tf.reduce_sum(x_target, axis=1)
            for i, k in enumerate(top_k):
                ideal_hits = tf.minimum(target_sum, k)
                idcg = tf.gather(rcpl_cumsums, tf.cast(ideal_hits, tf.int32) - 1)
                hits = tf.reduce_sum(hit_flags[:, :k], axis=1)
                dcg = tf.reduce_sum(hit_flags[:, :k] * rcpl_vals[:k], axis=1)
                rr = tf.reduce_max(hit_flags[:, :k] * rr_vals[:k], axis=1)
                ndcg_sum[i] += tf.reduce_sum(dcg / idcg)
                recall_sum[i] += tf.reduce_sum(hits / ideal_hits)
                mrr_sum[i] += tf.reduce_sum(rr)
        recall = recall_sum / total_size
        ndcg = ndcg_sum / total_size
        mrr = mrr_sum / total_size
        if return_dict:
            recall = {f'recall@{k}': v for k, v in zip(top_k, recall)}
            ndcg = {f'ndcg@{k}': v for k, v in zip(top_k, ndcg)}
            mrr = {f'mrr@{k}': v for k, v in zip(top_k, mrr)}
            return recall | ndcg | mrr
        return recall, ndcg, mrr

    def compile(
        self,
        optimizer=None,
        learning_rate=1e-3,
        weight_decay=0.0,
        **kwargs,
    ):
        if optimizer is None:
            optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        super().compile(optimizer=optimizer)

    def fit(
        self,
        ds,
        n_epochs=1000,
        batch_size=100,
        shuffle=True,
        callbacks=[],
        verbose=1,
        **kwargs,
    ):
        if shuffle:
            ds = ds.shuffle(buffer_size=ds.cardinality())
        if batch_size is not None:
            ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return super().fit(ds, epochs=n_epochs, callbacks=callbacks, verbose=verbose)
