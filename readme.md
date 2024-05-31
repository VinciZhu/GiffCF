# GiffCF

This is our TensorFlow implementation for the paper:

> Yunqin Zhu, Chao Wang, Qi Zhang, and Hui Xiong. 2024. Graph Signal Diffusion Model for Collaborative Filtering. In SIGIR ’24, July 14–18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 11 pages.

See the arXiv version at https://arxiv.org/abs/2311.08744. If you have any questions, please feel free to contact Yunqin Zhu (haaasined at gmail dot com) or create an issue.

## Introduction

GiffCF is a variant of diffusion model tailored for collaborative filtering with implicit feedback, which smooths and sharpens graph signals by simulating the heat equation. The hierarchical generative model leverages the adjacency of the item-item similarity graph and enables high-quality reconstruction of user-item interactions. It improves upon previous diffusion recommender models and graph signal processing techniques.

## Requirements

The code has been tested running under Python 3.11.5, on both RTX 3080 Ti & 3090 GPUs. The following packages are required:

```
numpy==1.24.3
scipy==1.13.0
tabulate==0.9.0
tensorflow==2.14.0
tomli==2.0.1
```

## Usage

### Installation

To download the code and install the dependencies:

```
git clone https://github.com/HasiNed/GiffCF
cd GiffCF
pip install -r requirements.txt
```

### Arguments

Run `python main.py -h` to show the help message:

```
usage: main.py [-h] [-c CONFIG_PATH] [-p PARAMETERS] [-v VERBOSE] {train,evaluate}

positional arguments:
  {train,evaluate}      whether to train or evaluate a GiffCF model

options:
  -h, --help            show this help message and exit
  -c CONFIG_PATH        path to config file in TOML format
  -p PARAMETERS         dict overriding config parameters
  -v VERBOSE, --verbose VERBOSE
```

We support both TOML files and command-line key-value pairs for configuration. You can start by overwriting the default config files in `./configs` or passing a Python dictionary of parameters to the script.

### Datasets

We borrow the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [Yelp](https://www.yelp.com/dataset/), and [Amazon-Book](https://jmcauley.ucsd.edu/data/amazon/) datasets from [DiffRec](https://github.com/YiyanXu/DiffRec), with the same training/validation/test splits. The datasets are preprocessed and stored in `./datasets`, each containing the following files:

- `train_list.npy`, `valid_list.npy`, `test_list.npy`: lists of user-item interactions in the format of `(user_id, item_id)`, for training, validation, and testing, respectively.
- `cache/eigen.npy`: truncated eigendecomposition of the normalized Laplacian matrix. Technically you can remove this file and let GiffCF compute and cache it on the fly. (Due to the limitation of repository size, we do not include this file for Amazon-Book.)

By default, our script selects the dataset by its name (`movielens`, `yelp`, or `amazon`) and loads from the pre-defined paths. You can also specify the dataset path in the TOML file like:

```
[dataset]
path = './datasets/ml-1m_clean'
```

or in the command line like `-p "{'dataset': {'path': './datasets/ml-1m_clean'}}"`.

### Inference

We provide two sets of checkpoints for all the three datasets:

- `./checkpoints/legacy/*.keras`: used in the original paper.
- `./checkpoints/*.keras`: trained with the same hyperparameters but an updated version of codebase, which yield slightly better results.

The default config files for inference are located in `./configs/inference`. For example, to run inference on the Amazon-Book dataset:
```
python main.py evaluate -c ./configs/inference/amazon.toml
```
To reproduce the exact results in the paper, you can specify the checkpoint path like:
```
python main.py evaluate -c ./configs/inference/amazon.toml -p "{'model': {'ckpt_path': './checkpoints/legacy/amazon.keras'}}"
```

The recommendation performance on the testing set should be identical to the following table:

| Checkpoint         | Recall@10  | Recall@20  | NDCG@10    | NDCG@20    | MRR@10     | MRR@20     |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| `legacy/movielens` | 0.1251     | 0.1947     | 0.0962     | 0.1221     | 0.1577     | 0.1704     |
| `legacy/yelp`      | 0.0649     | 0.1063     | 0.0397     | 0.0523     | 0.0476     | 0.0527     |
| `legacy/amazon`    | 0.1121     | 0.1528     | 0.0733     | 0.0858     | 0.0795     | 0.0842     |
| `movielens`        | **0.1308** | **0.2024** | **0.1002** | **0.1272** | **0.1632** | **0.1759** |
| `yelp`             | **0.0653** | **0.1063** | **0.0399** | **0.0524** | **0.0479** | **0.0530** |
| `amazon`           | **0.1122** | **0.1534** | **0.0734** | **0.0861** | **0.0796** | **0.0844** |

### Training

The default config files for training are located in `./configs` and produce the same results as the last three rows of the table above. For example, to train GiffCF on the MovieLens-1M dataset:
```
python main.py train -c ./configs/movielens.toml
```

To train on the Yelp dataset with a customized diffusion schedule:
```
python main.py train -c ./configs/yelp.toml -p "{'model': {'T': 2, 'alpha': 1.0, 'noise_scale': 0.1}}"
```

**Note**: While our denoiser architecture is designed to be expressive, it may suffer from some common training issues like local minima or overfitting. We recommend using a different random seed or learning rate schedule if you find the model underperforms on your own dataset. It remains an open question how to design a more robust training algorithm for GiffCF, and we welcome any suggestions or contributions. Good luck!

## Citation

If you use our code, please kindly cite:

```
@inproceedings{zhuGraphSignalDiffusion2024,
  title={Graph Signal Diffusion Model for Collaborative Filtering},
  author={Zhu, Yunqin and Wang, Chao and Zhang, Qi and Xiong, Hui},
  booktitle={SIGIR},
  year={2024}
}
```
