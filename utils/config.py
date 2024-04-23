import os
import ast
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # noqa: F401


def read_toml(path):
    with open(path, 'rb') as file:
        return tomllib.load(file)


def parse_toml(toml):
    return tomllib.loads(toml)


def dict_to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_namespace(v)
        setattr(ns, k, v)
    return ns


def namespace_to_dict(ns: SimpleNamespace) -> dict:
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            v = namespace_to_dict(v)
        d[k] = v
    return d


def merge_dict(d1: dict, d2: dict) -> dict:
    for k, v in d1.items():
        if k in d2 and isinstance(v, dict):
            merge_dict(v, d2[k])
            d2.pop(k)
    d1.update(d2)
    return d1


def list_configs(path):
    configs = []
    for root, _, files in os.walk(path):
        configs += [
            os.path.join(root, file) for file in files if file.endswith('.toml')
        ]
    configs = sorted(configs)
    return configs


def load_config(path, return_dict=False) -> SimpleNamespace:
    name = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path)
    config = {'name': name, **read_toml(path)}
    imports = config.pop('imports', [])
    parent_config = {}
    for parent in imports:
        parent_path = os.path.join(dirname, f'{parent}.toml')
        parent_config = merge_dict(
            parent_config, load_config(parent_path, return_dict=True)
        )
    config = merge_dict(parent_config, config)
    return config if return_dict else dict_to_namespace(config)


def parse_dict(src: str) -> dict:
    try:
        parsed_dict = ast.literal_eval(src)
        return parsed_dict
    except (ValueError, SyntaxError) as e:
        raise ValueError(f'Error parsing dictionary: {e}')
