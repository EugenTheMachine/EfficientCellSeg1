import os
import time
from pathlib import Path

from ray import tune

from data.utils import read_mask_to_numpy, resize_mask
from experiment_in_paper.ray.utils import load_configs
from metrics import average_precision
from project_root import PROJECT_ROOT


def objective(config):
    config = config["config"]
    pred_masks_folder = Path(config["result_pth_path"]).parent / "pred_masks"
    pred_masks_files = sorted(list(Path(pred_masks_folder).iterdir()))
    pred_masks = [read_mask_to_numpy(i) for i in pred_masks_files]
    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]
    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]
    return {"score": score, "ap": ap}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "8"
    select_dataset = [
        "cellpose_generalized",
        "cellpose_specialized",
        "cellseg_blood",
        "deepbacs_rod_brightfield",
        "deepbacs_rod_fluorescence",
        "dsb2018_stardist",
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
    select_train_num = None

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    configs_dict = load_configs(
        [
            PROJECT_ROOT / "experiment_in_paper/robustness/configs",
            PROJECT_ROOT / "experiment_in_paper/train_image_numbers/configs",
            PROJECT_ROOT / "experiment_in_paper/vit_size/configs",
            PROJECT_ROOT / "experiment_in_paper/batch_size/configs",
        ],
        select_dataset=select_dataset,
        select_train_num=select_train_num,
    )
    configs = list(configs_dict.values())

    search_space = {
        "config": tune.grid_search(configs),
    }

    tuner = tune.Tuner(
        trainable=tune.with_resources(objective, resources={"cpu": 1, "gpu": 1}),
        param_space=search_space,
    )

    results = tuner.fit()
    df = results.get_dataframe()
    df = df[
        [
            "score",
            "config/config/train_id",
            "config/config/dataset_name",
            "config/config/train_num",
            "config/config/vit_name",
            "config/config/epoch_max",
            "config/config/batch_size",
        ]
    ]
    df.rename(
        columns={
            "score": "ap_0.5",
            "config/config/train_id": "train_id",
            "config/config/dataset_name": "dataset_name",
            "config/config/train_num": "train_num",
            "config/config/vit_name": "vit_name",
            "config/config/epoch_max": "epoch_max",
            "config/config/batch_size": "batch_size",
        },
        inplace=True,
    )
    df = df[["dataset_name", "train_num", "train_id", "ap_0.5", "vit_name", "epoch_max", "batch_size"]]
    df.to_csv(PROJECT_ROOT / f"experiment_in_paper/result/result_{time_str}.csv", index=False)
