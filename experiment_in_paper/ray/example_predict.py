import os
from pathlib import Path

import yaml

from data.utils import read_mask_to_numpy, resize_mask
from metrics import average_precision
from predict import predict_config
from project_root import PROJECT_ROOT

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = (
        PROJECT_ROOT
        / "experiment_in_paper/robustness/configs/cellseg1_cellseg_blood_117.yaml"
    )
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["result_pth_path"] = str(PROJECT_ROOT / "checkpoints/cellseg_blood_117.pth")
    # suppose the data is at /data2/zhoupeilin/celldata/cellseg_blood/test/images
    config["data_dir"] = "/data2/zhoupeilin/celldata/cellseg_blood"
    # the predicted masks will be saved here
    config["result_dir"] = (
        "/data2/zhoupeilin/celldata/cellseg_blood/cellseg1/robustness/cellseg1_cellseg_blood_117"
    )

    pred_masks = predict_config(config, save=True)

    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]
    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]
    print(f"mAP@0.5: {score}")
