import shutil
import yaml
from project_root import PROJECT_ROOT
from set_environment import set_env

if __name__ == "__main__":
    data_statistics_file = str(PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml")
    yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/batch_size/configs"
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
        # "cellpose_specialized",
        # "cellseg_blood",
        # "deepbacs_rod_brightfield",
        # "deepbacs_rod_fluorescence",
        # "dsb2018_stardist",
    ]

    # Only select the first few dataset ids to reduce the number of trials
    # select_id = [60, 138, 6, 70, 435][:5]  # Choose 20 ids for simplification

    for i, dataset_name in enumerate(select_dataset):
        dataset_now = datasets[dataset_name]

        # Use the first 20 unique IDs for trials to reduce the number of trials.
        for batch_size in [2]:
            with open(example_config) as f:
                config = yaml.safe_load(f)

            # config["batch_size"] = batch_size
            # config["gradient_accumulation_step"] = int(32 / batch_size)
            config["track_gpu_memory"] = True

            config["method_name"] = "cellseg1"
            config["data_dir"] = dataset_now["data_dir"]
            config["dataset_name"] = dataset_name

            # Each config name will be unique and identifiable
            config_name = (
                f"{config['method_name']}_{config['dataset_name']}_train_id_{i}_batch_size_{batch_size}"
            )

            # config["resize_size"] = dataset_now["resize_size"]
            # config["patch_size"] = 256
            # config["crop_n_layers"] = 1

            # config["train_num"] = "full"  # Train on the full dataset
            # config["train_id"] = None  # Use all images for training
            # config["epoch_max"] = 20  # Train for 20 epochs

            # config["result_dir"] = f"{dataset_now['data_dir']}/cellseg1/batch_size/{config_name}"
            # config["train_image_dir"] = f"{config['data_dir']}/train/images"
            # config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
            # config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

            yaml_path = yaml_save_dir / f"{config_name}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(config, f)





















# import shutil

# import yaml

# from project_root import PROJECT_ROOT
# from set_environment import set_env

# if __name__ == "__main__":
#     data_statistics_file = str(PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml")
#     yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/batch_size/configs"
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

#     select_id = [60, 138, 6, 70, 435]

#     for i, dataset_name in enumerate(select_dataset):
#         dataset_now = datasets[dataset_name]
#         train_id = select_id[i]

#         for batch_size in [1, 2, 4]:
#             with open(example_config) as f:
#                 config = yaml.safe_load(f)

#             config["batch_size"] = batch_size
#             config["gradient_accumulation_step"] = int(32 / batch_size)
#             config["track_gpu_memory"] = True

#             config["method_name"] = "cellseg1"
#             config["data_dir"] = dataset_now["data_dir"]
#             config["dataset_name"] = dataset_name
#             config_name = (
#                 f"{config['method_name']}_{config['dataset_name']}_train_id_{train_id}_batch_size_{batch_size}"
#             )

#             config["resize_size"] = dataset_now["resize_size"]
#             config["patch_size"] = 256
#             config["crop_n_layers"] = 1

#             config["train_num"] = 1
#             config["train_id"] = [train_id]

#             config["result_dir"] = f"{dataset_now['data_dir']}/cellseg1/batch_size/{config_name}"
#             config["train_image_dir"] = f"{config['data_dir']}/train/images"
#             config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
#             config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

#             yaml_path = yaml_save_dir / f"{config_name}.yaml"
#             with open(yaml_path, "w") as f:
#                 yaml.dump(config, f)
