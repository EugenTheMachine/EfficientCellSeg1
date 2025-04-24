import os

import yaml

from cellseg1_train import main
from project_root import PROJECT_ROOT

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = (
        PROJECT_ROOT
        / "experiment_in_paper/robustness/configs/cellseg1_cellseg_blood_117.yaml"
    )
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["train_image_dir"] = "/data2/zhoupeilin/celldata/cellseg_blood/train/images"
    config["train_mask_dir"] = "/data2/zhoupeilin/celldata/cellseg_blood/train/masks"
    # the result lora will be saved here
    config["result_pth_path"] = str(PROJECT_ROOT / "checkpoints/cellseg_blood_117.pth")
    model = main(config, save_model=True)
