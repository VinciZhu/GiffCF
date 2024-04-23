import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense


class Timestep(Layer):
    def __init__(self, embed_dim, n_steps, max_wavelength=10000.0):
        assert embed_dim % 2 == 0
        super().__init__()
        timescales = np.power(max_wavelength, -np.arange(0, embed_dim, 2) / embed_dim)
        timesteps = np.arange(n_steps + 1)
        angles = timesteps[:, np.newaxis] * timescales[np.newaxis, :]
        sinusoids = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        self.sinusoids = tf.constant(sinusoids, dtype=tf.float32)

    def call(self, timesteps):
        return tf.gather(self.sinusoids, tf.reshape(timesteps, [-1]))


class TimeEmbed(Layer):
    def __init__(self, hidden_dim, out_dim, activation, n_steps):
        super().__init__()
        self.timestep = Timestep(hidden_dim, n_steps)
        self.hidden = Dense(hidden_dim, activation)
        self.out = Dense(out_dim)

    def call(self, t):
        e = self.timestep(t)
        return self.out(self.hidden(e))


class SimpleMixer(Layer):
    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.hidden = Dense(hidden_dim, activation)
        self.out = Dense(1)

    def call(self, inputs):
        x = tf.stack(inputs, axis=-1)
        x = self.out(self.hidden(x))
        return tf.squeeze(x, axis=-1)


class Denoiser(Layer):
    def __init__(
        self,
        n_items,
        embed_dim=200,
        activation='swish',
        initializer='glorot_uniform',
        norm_ord=1,
        n_steps=10,
        ablation=None,
    ):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.norm_ord = norm_ord
        self.item_embed = self.add_weight(
            'item_embed',
            shape=(n_items, embed_dim),
            initializer=initializer,
            trainable=True,
        )
        self.t_embed1 = TimeEmbed(20, 1, activation, n_steps)
        self.t_embed2 = TimeEmbed(20, 1, activation, n_steps)
        self.embed_mixer = SimpleMixer(2, activation)
        self.score_mixer = SimpleMixer(2, activation)
        self.ablation = ablation

    @property
    def mlp_weights(self):
        return (
            self.t_embed1.trainable_weights
            + self.t_embed2.trainable_weights
            + self.embed_mixer.trainable_weights
            + self.score_mixer.trainable_weights
        )

    def norm(self, c):
        if self.norm_ord is not None:
            norm = tf.norm(c, ord=self.norm_ord, axis=-1, keepdims=True)
            return tf.maximum(norm, 1.0)
        else:
            return 1.0

    def call(self, z_t, c, Ac, t, training=None):
        t_embed1 = tf.repeat(self.t_embed1(t), self.embed_dim, axis=1)
        t_embed2 = tf.repeat(self.t_embed2(t), self.n_items, axis=1)
        z_embed = tf.matmul(z_t / self.n_items, self.item_embed)
        c_embed = tf.matmul(c / self.norm(c), self.item_embed)
        if self.ablation == 'wo_latent':
            x_embed = self.embed_mixer([c_embed, t_embed1])
        elif self.ablation == 'wo_precond':
            x_embed = self.embed_mixer([z_embed, t_embed1])
        else:
            x_embed = self.embed_mixer([z_embed, c_embed, t_embed1])
        x_mid = tf.matmul(x_embed, self.item_embed, transpose_b=True)
        if self.ablation == 'wo_postcond':
            x_pred = self.score_mixer([x_mid, t_embed2])
        else:
            x_pred = self.score_mixer([x_mid, c, Ac, t_embed2])
        return x_pred
