import random
import shutil
from pathlib import Path
import yaml

from project_root import PROJECT_ROOT
from set_environment import set_env

if __name__ == "__main__":
    data_statistics_file = str(PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml")
    yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/robustness/configs"
    example_config = str(PROJECT_ROOT / "example_config.yaml")

    with open(example_config) as f:
        config = yaml.safe_load(f)
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )

    with open(data_statistics_file) as f:
        datasets = yaml.safe_load(f)["dataset"]
    if yaml_save_dir.exists():
        shutil.rmtree(yaml_save_dir)
    yaml_save_dir.mkdir(exist_ok=True)

    select_dataset = [
        "livecell",
    ]

    for k, dataset in datasets.items():
        if k not in select_dataset:
            continue
        # if k == "dsb2018_stardist":
        #     train_ids = sorted(random.sample(list(range(448)), 100))
        # else:
        #     # Select only 20 images to generate configurations
        #     train_ids = random.sample(list(range(dataset["train_image_num"])), 5)

        for train_id in [20]:
            with open(example_config) as f:
                config = yaml.safe_load(f)

            config["method_name"] = "cellseg1"
            config["data_dir"] = dataset["data_dir"]
            config["dataset_name"] = Path(config["data_dir"]).stem
            config_name = f"{config['method_name']}_{config['dataset_name']}_{train_id}"

            # config["resize_size"] = dataset["resize_size"]
            # config["patch_size"] = 256
            # config["crop_n_layers"] = 1

            # config["train_num"] = "full"  # Train on the full dataset
            # config["train_id"] = None  # Use all images for training
            # config["epoch_max"] = 20  # Train for 20 epochs

            # config["result_dir"] = f"{dataset['data_dir']}/cellseg1/robustness/{config_name}"
            # config["train_image_dir"] = f"{config['data_dir']}/train/images"
            # config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
            # config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

            yaml_path = yaml_save_dir / f"{config_name}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(config, f)




















# import random
# import shutil
# from pathlib import Path

# import yaml

# from project_root import PROJECT_ROOT
# from set_environment import set_env

# if __name__ == "__main__":
#     data_statistics_file = str(PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml")
#     yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/robustness/configs"
#     example_config = str(PROJECT_ROOT / "example_config.yaml")

#     with open(example_config) as f:
#         config = yaml.safe_load(f)
#     set_env(
#         config["deterministic"],
#         config["seed"],
#         config["allow_tf32_on_cudnn"],
#         config["allow_tf32_on_matmul"],
#     )

#     with open(data_statistics_file) as f:
#         datasets = yaml.safe_load(f)["dataset"]
#     if yaml_save_dir.exists():
#         shutil.rmtree(yaml_save_dir)
#     yaml_save_dir.mkdir(exist_ok=True)

#     select_dataset = [
#         "livecell",
#         # "cellpose_specialized",
#         # "cellseg_blood",
#         # "deepbacs_rod_brightfield",
#         # "deepbacs_rod_fluorescence",
#         # "dsb2018_stardist",
#     ]

#     for k, dataset in datasets.items():
#         if k not in select_dataset:
#             continue
#         if k == "dsb2018_stardist":
#             train_ids = sorted(random.sample(list(range(448)), 100))
#         else:
#             train_ids = list(range(dataset["train_image_num"]))
#         for train_id in train_ids:
#             with open(example_config) as f:
#                 config = yaml.safe_load(f)

#             config["method_name"] = "cellseg1"
#             config["data_dir"] = dataset["data_dir"]
#             config["dataset_name"] = Path(config["data_dir"]).stem
#             config_name = f"{config['method_name']}_{config['dataset_name']}_{train_id}"

#             config["resize_size"] = dataset["resize_size"]
#             config["patch_size"] = 256
#             config["crop_n_layers"] = 1

#             config["train_num"] = 1
#             config["train_id"] = [train_id]
#             # config["epoch_max"] = 50

#             config["result_dir"] = f"{dataset['data_dir']}/cellseg1/robustness/{config_name}"
#             config["train_image_dir"] = f"{config['data_dir']}/train/images"
#             config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
#             config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

#             yaml_path = yaml_save_dir / f"{config_name}.yaml"
#             with open(yaml_path, "w") as f:
#                 yaml.dump(config, f)
