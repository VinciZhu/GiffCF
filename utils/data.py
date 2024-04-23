import os
import numpy as np
import tensorflow as tf
from scipy import sparse as sp

from typing import Literal, Union


def load_diffrec_data(path):
    train_ids = np.load(os.path.join(path, 'train_list.npy'))
    val_ids = np.load(os.path.join(path, 'valid_list.npy'))
    test_ids = np.load(os.path.join(path, 'test_list.npy'))
    x_train = sp.csr_array((np.ones(len(train_ids), np.float32), train_ids.T))
    x_val = sp.csr_array(
        (np.ones(len(val_ids), np.float32), val_ids.T), shape=x_train.shape
    )
    x_test = sp.csr_array(
        (np.ones(len(test_ids), np.float32), test_ids.T), shape=x_train.shape
    )
    return x_train, x_val, x_test


def save_lightgcn_data(x_train, x_val, x_test, path):
    def save_adj_list(sp_mat, path):
        sp_mat = sp_mat.tocoo()
        n_users = sp_mat.shape[0]
        data = {user: [] for user in range(n_users)}
        for row, col, _ in zip(sp_mat.row, sp_mat.col, sp_mat.data):
            data[row].append(col)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for user, neighbors in data.items():
                neighbors = ' '.join(map(str, neighbors))
                f.write(f'{user} {neighbors}\n')

    save_adj_list(x_train, os.path.join(path, 'train.txt'))
    save_adj_list(x_val, os.path.join(path, 'val.txt'))
    save_adj_list(x_test, os.path.join(path, 'test.txt'))


def create_datasets(
    x_train, x_val, x_test, batch_size=100, test_batch_size=400, **kwargs
):
    x_val_history = x_test_history = array_to_dataset(x_train)
    x_val_target = array_to_dataset(x_val)
    x_test_target = array_to_dataset(x_test)
    val_mask = array_to_dataset(x_train > 0)
    test_mask = array_to_dataset(x_train + x_val > 0)

    train_ds = array_to_dataset(x_train)
    val_ds = tf.data.Dataset.zip((x_val_history, x_val_target, val_mask))
    test_ds = tf.data.Dataset.zip((x_test_history, x_test_target, test_mask))
    return train_ds, val_ds, test_ds


def array_to_dataset(x):
    if isinstance(x, sp.csr_array):
        x = sparse_array_to_tensor(x)
    else:
        x = tf.convert_to_tensor(x)
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(x)
    if isinstance(x, tf.SparseTensor):
        dataset = dataset.map(lambda x: tf.sparse.to_dense(x))
    return dataset


def sparse_array_to_tensor(x: sp.csr_array):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    x = tf.SparseTensor(indices, coo.data, coo.shape)
    return tf.sparse.reorder(x)


# Unused
def get_adjacency_matrix(
    rating: Union[np.array, sp.csr_array],
    sparse: bool = False,
    return_tensor: bool = False,
):
    n_users, n_items = rating.shape
    adjacency = sp.lil_matrix((n_users + n_items, n_users + n_items))
    adjacency[:n_users, n_users:] = rating
    adjacency[n_users:, :n_users] = rating.T
    if return_tensor:
        adjacency = sparse_array_to_tensor(adjacency)
        return adjacency if sparse else tf.sparse.to_dense(adjacency)
    else:
        adjacency = adjacency.tocsr()
        return adjacency if sparse else adjacency.toarray()


# Unused
def normalize_matrix(mat: np.array, method: Literal['left', 'right', 'sym'] = 'sym'):
    row_sum = mat.sum(axis=1).flatten()
    col_sum = mat.sum(axis=0).flatten()
    with np.errstate(divide='ignore'):
        if method == 'left':
            mat = mat / row_sum[:, np.newaxis]
        elif method == 'right':
            mat = mat / col_sum[np.newaxis, :]
        elif method == 'sym':
            mat = mat / np.sqrt(row_sum[:, np.newaxis] * col_sum[np.newaxis, :])
        else:
            raise ValueError('Invalid value for method.')
    return mat


# Unused
def normalize_sparse_matrix(
    sp_mat: sp.csr_array,
    method: Literal['left', 'right', 'sym'] = 'sym',
    return_tensor: bool = False,
):
    row_sum = sp_mat.sum(axis=1).flatten()
    col_sum = sp_mat.sum(axis=0).flatten()
    with np.errstate(divide='ignore'):
        left = sp.diags(1 / row_sum)
        right = sp.diags(1 / col_sum)
    if method == 'left':
        sp_mat = left @ sp_mat
    elif method == 'right':
        sp_mat = sp_mat @ right
    elif method == 'sym':
        sp_mat = np.sqrt(left) @ sp_mat @ np.sqrt(right)
    else:
        raise ValueError('Invalid value for method.')
    return sparse_array_to_tensor(sp_mat) if return_tensor else sp_mat
