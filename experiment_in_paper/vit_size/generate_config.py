import shutil

import yaml

from project_root import PROJECT_ROOT
from set_environment import set_env

if __name__ == "__main__":
    data_statistics_file = str(PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml")
    yaml_save_dir = PROJECT_ROOT / "experiment_in_paper/vit_size/configs"
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

    # select_id = [60, 138, 6, 70, 435]
    sam_models = {
        "vit_h": PROJECT_ROOT / "streamlit_storage" / "sam_backbone" / "sam_vit_h_4b8939.pth",
        "vit_l": PROJECT_ROOT / "streamlit_storage" / "sam_backbone" / "sam_vit_l_0b3195.pth",
        "vit_b": PROJECT_ROOT / "streamlit_storage" / "sam_backbone" / "sam_vit_b_01ec64.pth",
    }
    for i, dataset_name in enumerate(select_dataset):
        dataset_now = datasets[dataset_name]
        # train_id = select_id[i]

        # for vit_size in [
        #     "vit_b",
        #     "vit_l",
        #     "vit_h"
        #     ]:
        with open(example_config) as f:
            config = yaml.safe_load(f)

        config["method_name"] = "cellseg1"
        config["data_dir"] = dataset_now["data_dir"]
        config["dataset_name"] = dataset_name
        config_name = f"{config['method_name']}_{config['dataset_name']}_train_id_-_vit_size"

        # config["resize_size"] = dataset_now["resize_size"]
        # config["patch_size"] = 256
        # config["crop_n_layers"] = 1

        # config["train_num"] = 1
        # config["train_num"] = "full"
        # config["train_id"] = [train_id]
        # config["train_id"] = None
        # config["vit_name"] = vit_size
        # config["model_path"] = str(sam_models[vit_size])

        # config["result_dir"] = f"{dataset_now['data_dir']}/cellseg1/vit_size/{config_name}"
        # config["train_image_dir"] = f"{config['data_dir']}/train/images"
        # config["train_mask_dir"] = f"{config['data_dir']}/train/masks"
        # config["result_pth_path"] = f"{config['result_dir']}/sam_lora.pth"

        yaml_path = yaml_save_dir / f"{config_name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
