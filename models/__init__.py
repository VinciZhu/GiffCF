import os

from .giff_cf import GiffCF
from .recommender import Recommender


def load_model(name, ckpt_path=None, input=None, **kwargs) -> Recommender:
    if name == 'GiffCF':
        ModelClass = GiffCF
    else:
        raise NotImplementedError(f'Unknown model: {name}')
    model = ModelClass(interaction=input, n_items=input.shape[1], **kwargs)
    model(input[:1].toarray())  # Build model
    if ckpt_path is not None and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
    return model
