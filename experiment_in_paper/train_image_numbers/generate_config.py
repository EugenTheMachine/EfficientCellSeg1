import shutil
from pathlib import Path
import yaml

from project_root import PROJECT_ROOT
from set_environment import set_env

if __name__ == "__main__":
    data_statistics_file = str(
        PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml"
    )
    yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/train_image_numbers/configs"
    example_config = str(PROJECT_ROOT / "example_config.yaml")

    with open(example_config) as f:
        config = yaml.safe_load(f)
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )

    select_dataset = [
        "livecell",
    ]
    dataset_1 = [
        "tissuenet_Breast_20191211_IMC_nuclei",
        "tissuenet_Breast_20200116_DCIS_nuclei",
        "tissuenet_Breast_20200526_COH_BC_nuclei",
        "tissuenet_Epidermis_20200226_Melanoma_nuclei",
        "tissuenet_Epidermis_20200623_sizun_epidermis_nuclei",
        "tissuenet_GI_20191219_Eliot_nuclei",
        "tissuenet_GI_20200219_Roshan_nuclei",
        "tissuenet_GI_20200627_CODEX_CRC_nuclei",
        "tissuenet_Lung_20200210_CyCIF_Lung_LN_nuclei",
        "tissuenet_Lymph_Node_20200114_cHL_nuclei",
        "tissuenet_Lymph_Node_20200520_HIV_nuclei",
        "tissuenet_Pancreas_20200512_Travis_PDAC_nuclei",
        "tissuenet_Pancreas_20200624_CODEX_Panc_nuclei",
        "tissuenet_Tonsil_20200211_CyCIF_Tonsil_nuclei",
    ]

    with open(data_statistics_file) as f:
        all_datasets = yaml.safe_load(f)["dataset"]
    if yaml_save_dir.exists():
        shutil.rmtree(yaml_save_dir)
    yaml_save_dir.mkdir(exist_ok=True)

    for k, dataset in all_datasets.items():
        # if k in select_dataset:
        #     train_num_list = ["full"]  # Reduce the number of trials here
        # elif k == "cellpose_generalized":
        #     train_num_list = ["full"]
        # elif k in dataset_1:
        #     train_num_list = ["1"]
        # else:
        #     continue

        # Select only all images to generate configurations
        selected_image_indices = range(len(dataset["data_dir"]))
        train_num_list = ["full"]

        for train_num in train_num_list:
            # for idx in selected_image_indices:
                with open(example_config) as f:
                    config = yaml.safe_load(f)

                config["method_name"] = "cellseg1"
                config["data_dir"] = dataset["data_dir"]
                config["dataset_name"] = Path(config["data_dir"]).stem
                config_name = (
                    f"{config['method_name']}_{config['dataset_name']}_{train_num}_{0}"
                )

                # config["resize_size"] = dataset["resize_size"]
                # config["patch_size"] = 256
                # config["crop_n_layers"] = 1

                # config["train_num"] = train_num
                # config["train_id"] = [idx]
                # config["train_id"] = None

                # config["result_dir"] = (
                #     f"{dataset['data_dir']}/cellseg1/train_image_numbers/{config_name}"
                # )
                # config["train_image_dir"] = f"{config['data_dir']}/train/images"
                # config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
                # config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

                yaml_path = yaml_save_dir / f"{config_name}.yaml"
                with open(yaml_path, "w") as f:
                    yaml.dump(config, f)










# import shutil
# from pathlib import Path

# import yaml

# from project_root import PROJECT_ROOT
# from set_environment import set_env

# if __name__ == "__main__":
#     data_statistics_file = str(
#         PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml"
#     )
#     yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/train_image_numbers/configs"
#     example_config = str(PROJECT_ROOT / "example_config.yaml")

#     with open(example_config) as f:
#         config = yaml.safe_load(f)
#     set_env(
#         config["deterministic"],
#         config["seed"],
#         config["allow_tf32_on_cudnn"],
#         config["allow_tf32_on_matmul"],
#     )

#     select_dataset = [
#         "livecell",
#         # "cellpose_specialized",
#         # "cellseg_blood",
#         # "deepbacs_rod_brightfield",
#         # "deepbacs_rod_fluorescence",
#         # "dsb2018_stardist",
#     ]
#     dataset_1 = [
#         "tissuenet_Breast_20191211_IMC_nuclei",
#         "tissuenet_Breast_20200116_DCIS_nuclei",
#         "tissuenet_Breast_20200526_COH_BC_nuclei",
#         "tissuenet_Epidermis_20200226_Melanoma_nuclei",
#         "tissuenet_Epidermis_20200623_sizun_epidermis_nuclei",
#         "tissuenet_GI_20191219_Eliot_nuclei",
#         "tissuenet_GI_20200219_Roshan_nuclei",
#         "tissuenet_GI_20200627_CODEX_CRC_nuclei",
#         "tissuenet_Lung_20200210_CyCIF_Lung_LN_nuclei",
#         "tissuenet_Lymph_Node_20200114_cHL_nuclei",
#         "tissuenet_Lymph_Node_20200520_HIV_nuclei",
#         "tissuenet_Pancreas_20200512_Travis_PDAC_nuclei",
#         "tissuenet_Pancreas_20200624_CODEX_Panc_nuclei",
#         "tissuenet_Tonsil_20200211_CyCIF_Tonsil_nuclei",
#     ]

#     with open(data_statistics_file) as f:
#         all_datasets = yaml.safe_load(f)["dataset"]
#     if yaml_save_dir.exists():
#         shutil.rmtree(yaml_save_dir)
#     yaml_save_dir.mkdir(exist_ok=True)

#     for k, dataset in all_datasets.items():
#         if k in select_dataset:
#             train_num_list = ["5", "10", "full"]
#         elif k == "cellpose_generalized":
#             train_num_list = ["full"]
#         elif k == "livecell":  # assuming that livecell dataset is full
#             # train_num_list = ["full"]
#             train_num_list = ["1"]
#         elif k in dataset_1:
#             train_num_list = ["1"]
#         else:
#             continue
#         for train_num in train_num_list:
#             with open(example_config) as f:
#                 config = yaml.safe_load(f)

#             config["method_name"] = "cellseg1"
#             config["data_dir"] = dataset["data_dir"]
#             config["dataset_name"] = Path(config["data_dir"]).stem
#             config_name = (
#                 f"{config['method_name']}_{config['dataset_name']}_{train_num}"
#             )

#             config["resize_size"] = dataset["resize_size"]
#             config["patch_size"] = 256
#             config["crop_n_layers"] = 1

#             config["train_num"] = train_num
#             if train_num == "1":
#                 config["train_id"] = [0]
#             elif train_num == "5":
#                 config["train_id"] = list(range(5))
#             elif train_num == "10":
#                 config["train_id"] = list(range(10))
#             elif train_num == "full":
#                 config["train_id"] = None

#             # if train_num == "full":
#             #     if config["dataset_name"] == "deepbacs_rod_brightfield":
#             #         # config["epoch_max"] = 100
#             #         config["epoch_max"] = 50
#             #     else:
#             #         # config["epoch_max"] = 30
#             #         config["epoch_max"] = 50
#             # else:
#             #     # config["epoch_max"] = 300
#             #     config["epoch_max"] = 50

#             config["result_dir"] = (
#                 f"{dataset['data_dir']}/cellseg1/train_image_numbers/{config_name}"
#             )
#             config["train_image_dir"] = f"{config['data_dir']}/train/images"
#             config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
#             config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

#             yaml_path = yaml_save_dir / f"{config_name}.yaml"
#             with open(yaml_path, "w") as f:
#                 yaml.dump(config, f)
