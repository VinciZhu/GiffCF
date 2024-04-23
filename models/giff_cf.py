import os
import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import svds

from keras.metrics import Mean
from keras.optimizers import Adam

from models.denoiser import Denoiser
from models.recommender import Recommender
from utils import sparse_array_to_tensor


class GiffCF(Recommender):
    def __init__(
        self,
        interaction,
        cache_path=None,
        embed_dim=200,
        activation='swish',
        initializer='glorot_uniform',
        dropout=0.5,
        norm_ord=1,
        T=3,
        t=None,
        alpha=1.5,
        ideal_weight=0.0,
        ideal_cutoff=200,
        noise_decay=1.0,
        noise_scale=0.0,
        ablation=None,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        n_items = interaction.shape[1]
        self.denoiser = Denoiser(
            n_items, embed_dim, activation, initializer, norm_ord, T, ablation
        )
        user_deg = interaction.sum(axis=1)[:, np.newaxis]
        item_deg = interaction.sum(axis=0)[np.newaxis, :]
        adj_right = user_deg ** (-1 / 4) * interaction * item_deg ** (-1 / 2)
        self.adj_right = sparse_array_to_tensor(adj_right.astype(np.float32))
        self.adj_left = sparse_array_to_tensor(adj_right.T.astype(np.float32))
        self.ideal_weight = ideal_weight
        if self.ideal_weight == 0.0:
            ideal_cutoff = 1
        if cache_path is None:
            eigen = compute_eigen(adj_right, ideal_cutoff)
        else:
            cache_file = os.path.join(cache_path, 'eigen.npy')
            eigen = compute_eigen_with_cache(adj_right, ideal_cutoff, cache_file)
        self.eigen_val = tf.constant(eigen['values'], dtype=tf.float32)
        self.eigen_vec = tf.constant(eigen['vectors'], dtype=tf.float32)
        self.T = T
        self.t = np.linspace(0, T, T + 1, dtype=np.int32) if t is None else t
        self.alpha = alpha
        self.noise_decay = noise_decay
        self.noise_scale = noise_scale
        self.loss_tracker = Mean(name='loss')

    def prop(self, x):
        x_prop = tf.sparse.sparse_dense_matmul(self.adj_right, x, adjoint_b=True)
        x_prop = tf.sparse.sparse_dense_matmul(self.adj_left, x_prop)
        x_prop = tf.transpose(x_prop)
        return x_prop / self.eigen_val[0]

    def ideal(self, x, cutoff=None):
        eigen_vec = self.eigen_vec[:cutoff] if cutoff is not None else self.eigen_vec
        x_ideal = tf.matmul(x, eigen_vec, transpose_b=True)
        x_ideal = tf.matmul(x_ideal, eigen_vec)
        return x_ideal

    def smooth(self, x):
        if self.ideal_weight:
            x_smooth = self.prop(x) + self.ideal_weight * self.ideal(x)
            return x_smooth / (1 + self.ideal_weight)
        else:
            return self.prop(x)

    def filter(self, x, Ax, t):
        t = tf.cast(t, tf.float32)
        return x + self.alpha * t / self.T * (Ax - x)

    def sigma(self, t):
        t = tf.cast(t, tf.float32)
        return self.noise_scale * self.noise_decay ** (self.T - t)

    def denoise(self, z_t, c, Ac, t, training=False):
        t = tf.broadcast_to(t, (tf.shape(z_t)[0], 1))
        x_pred = self.denoiser(z_t, c, Ac, t, training)
        return x_pred

    def call(self, x, training=None):
        if training:
            t = tf.random.uniform((tf.shape(x)[0], 1), 1, self.T + 1, dtype=tf.int32)
            Ax = self.smooth(x)
            z_t = self.filter(x, Ax, t)
            if self.noise_scale > 0.0:
                eps = tf.random.normal(tf.shape(x))
                z_t += self.sigma(t) * eps
            c = tf.nn.dropout(x, self.dropout)
            Ac = self.smooth(c)
            x_pred = self.denoise(z_t, c, Ac, t, training)
            return x_pred
        else:
            Ax = self.smooth(x)
            z_t = self.filter(x, Ax, self.t[-1])
            for i in range(len(self.t) - 1, 0, -1):
                t, s = self.t[i], self.t[i - 1]
                x_pred = self.denoise(z_t, x, Ax, t)
                Ax_pred = self.smooth(x_pred)
                z_s_pred = self.filter(x_pred, Ax_pred, s)
                if self.noise_decay > 0.0:
                    z_t_pred = self.filter(x_pred, Ax_pred, t)
                    z_t = z_s_pred + self.noise_decay ** (t - s) * (z_t - z_t_pred)
                else:
                    z_t = z_s_pred
            x_pred = z_t
            return x_pred

    def compile(
        self,
        optimizers=[None, None],
        learning_rates=[1e-3, 1e-3],
        weight_decays=[0.0, 0.0],
        **kwargs,
    ):
        super().compile()
        self.optimizers = []
        for optim, lr, wd in zip(optimizers, learning_rates, weight_decays):
            if optim is None:
                optim = Adam(learning_rate=lr, weight_decay=wd)
            self.optimizers.append(optim)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = tf.reduce_sum(tf.square(x - x_pred), axis=1)
            reduced_loss = tf.reduce_mean(loss)
        weights = [self.denoiser.item_embed] + self.denoiser.mlp_weights
        grads = tape.gradient(reduced_loss, weights)
        self.optimizers[0].apply_gradients([(grads[0], weights[0])])
        self.optimizers[1].apply_gradients(zip(grads[1:], weights[1:]))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}


def compute_eigen(adj_right, cutoff):
    _, values, vectors = svds(adj_right, k=cutoff)
    idx = np.argsort(values)[::-1]
    values = values[idx] ** 2
    vectors = vectors[idx]
    return {'cutoff': cutoff, 'values': values, 'vectors': vectors}


def compute_eigen_with_cache(adj_right, cutoff, cache_file):
    if os.path.exists(cache_file):
        eigen = np.load(cache_file, allow_pickle=True).item()
        if eigen['cutoff'] >= cutoff:
            eigen['values'] = eigen['values'][:cutoff]
            eigen['vectors'] = eigen['vectors'][:cutoff]
            return eigen
    eigen = compute_eigen(adj_right, cutoff)
    np.save(cache_file, eigen)
    return eigen
