from pathlib import Path

import yaml


def load_configs(
    config_dirs, select_method=None, select_dataset=None, select_train_num=None
):
    if select_train_num is not None:
        select_train_num = [str(i) for i in select_train_num]
    if isinstance(config_dirs, str) or isinstance(config_dirs, Path):
        config_dirs = [config_dirs]
    config_file_list = []
    for config_dir in config_dirs:
        config_file_list.extend(sorted(list(Path(config_dir).glob("*.yaml"))))

    configs = {}
    for config_file in config_file_list:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        method_name = config["method_name"]
        dataset_name = config["dataset_name"]
        train_num = config["train_num"]

        if select_method is not None:
            if method_name not in select_method:
                continue
        if select_dataset is not None:
            if dataset_name not in select_dataset:
                continue
        if select_train_num is not None:
            if str(train_num) not in select_train_num:
                continue
        configs[f"{config_file.stem}_{config['train_num']}"] = config
    return configs
