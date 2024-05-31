import os
import time
import shutil
import argparse
from pprint import pprint
from tabulate import tabulate

from utils import (
    load_config,
    parse_dict,
    merge_dict,
    dict_to_namespace,
    seed_everything,
    init_tensorflow,
    load_diffrec_data,
    create_datasets,
)
from models import load_model
from callbacks import create_callbacks


DEFAULT_DATA_PATHS = {
    'movielens': 'datasets/ml-1m_clean',
    'amazon': 'datasets/amazon-book_clean',
    'yelp': 'datasets/yelp_clean',
}


def train(cfg):
    init_tensorflow(verbose=cfg.verbose)
    # Load dataset
    if not hasattr(cfg.dataset, 'path'):
        cfg.dataset.path = DEFAULT_DATA_PATHS[cfg.dataset.name]
    x_train, x_val, x_test = load_diffrec_data(cfg.dataset.path)
    train_ds, val_ds, test_ds = create_datasets(
        x_train, x_val, x_test, **vars(cfg.model)
    )
    # Remove previous checkpoint
    if hasattr(cfg, 'output_dir'):
        shutil.rmtree(os.path.join(cfg.output_dir, cfg.name), ignore_errors=True)
    # Create model
    if not hasattr(cfg.model, 'cache_path'):
        cfg.model.cache_path = os.path.join(cfg.dataset.path, 'cache')
    seed_everything(cfg.model.seed)
    model = load_model(input=x_train, **vars(cfg.model))
    if hasattr(cfg.model, 'init_ckpt_path'):
        model.load_weights(cfg.model.init_ckpt_path)
    if cfg.verbose > 0:
        model.summary(expand_nested=True)
    # Create callbacks
    if not hasattr(cfg.model, 'ckpt_path'):
        cfg.model.ckpt_path = os.path.join(cfg.output_dir, cfg.name, 'checkpoint.keras')
    if not hasattr(cfg.model, 'log_dir'):
        cfg.model.log_dir = os.path.join(cfg.output_dir, cfg.name, 'logs')
    x_sample = None
    if hasattr(cfg.model, 'sample_users'):
        x_sample = x_train[: cfg.model.sample_users].toarray()
    callbacks = create_callbacks(val_ds, test_ds, x_sample=x_sample, **vars(cfg.model))
    # Train model
    model.compile(**vars(cfg.model))
    start_time = time.time()
    model.fit(train_ds, callbacks=callbacks, verbose=cfg.verbose, **vars(cfg.model))
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Load and test model
    model = load_model(input=x_train, **vars(cfg.model))
    res = model.evaluate(test_ds, return_dict=True, **vars(cfg.model))
    if cfg.verbose > 0:
        print(tabulate([(cfg.name, *res.values())], headers=res.keys()))
    if hasattr(cfg, 'output_dir'):
        with open(os.path.join(cfg.output_dir, cfg.name, 'results.txt'), 'w') as f:
            f.write(f'Training time: {elapsed_time:.2f} seconds\n')
            f.write(tabulate([(cfg.name, *res.values())], headers=res.keys()))
    return res


def evaluate(cfg):
    init_tensorflow(verbose=cfg.verbose)
    # Load dataset
    if not hasattr(cfg.dataset, 'path'):
        cfg.dataset.path = DEFAULT_DATA_PATHS[cfg.dataset.name]
    x_train, x_val, x_test = load_diffrec_data(cfg.dataset.path)
    train_ds, val_ds, test_ds = create_datasets(
        x_train, x_val, x_test, **vars(cfg.model)
    )
    # Load and test model
    if not hasattr(cfg.model, 'cache_path'):
        cfg.model.cache_path = os.path.join(cfg.dataset.path, 'cache')
    if not hasattr(cfg.model, 'ckpt_path'):
        cfg.model.ckpt_path = os.path.join(cfg.output_dir, cfg.name, 'checkpoint.keras')
    model = load_model(input=x_train, **vars(cfg.model))
    res = model.evaluate(test_ds, return_dict=True, **vars(cfg.model))
    if cfg.verbose > 0:
        print(tabulate([(cfg.name, *res.values())], headers=res.keys()))
    if hasattr(cfg, 'output_dir'):
        with open(os.path.join(cfg.output_dir, cfg.name, 'eval_results.txt'), 'w') as f:
            f.write(tabulate([(cfg.name, *res.values())], headers=res.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command',
        type=str,
        choices=['train', 'evaluate'],
        help='whether to train or evaluate a GiffCF model',
    )
    parser.add_argument(
        '-c',
        dest='cfg_path',
        metavar='CONFIG_PATH',
        type=str,
        help='path to config file in TOML format',
    )
    parser.add_argument(
        '-p',
        dest='params',
        metavar='PARAMETERS',
        type=parse_dict,
        help='dict overriding config parameters',
    )
    parser.add_argument('-v', '--verbose', type=int, default=1)
    args = parser.parse_args()
    cfg = dict()
    if args.cfg_path is not None:
        cfg = load_config(args.cfg_path, return_dict=True)
        if args.verbose > 0:
            print(f'Config loaded from {os.path.abspath(args.cfg_path)}')
    if args.params is not None:
        cfg = merge_dict(cfg, args.params)
    if args.verbose > 1:
        pprint(cfg, sort_dicts=False)
    cfg = dict_to_namespace(cfg)
    cfg.verbose = args.verbose
    if args.command == 'train':
        train(cfg)
    elif args.command == 'evaluate':
        evaluate(cfg)
