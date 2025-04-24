import os
import time
from pathlib import Path

from ray import tune

from data.utils import read_mask_to_numpy, resize_mask
from experiment_in_paper.ray.utils import load_configs
from metrics import average_precision
from project_root import PROJECT_ROOT


def objective(param):
    param = param["param"]
    config = param["config"]
    test_dataset_name = param["test_dataset_name"]
    test_data_dir = param["test_data_dir"]

    test_mask_folder = Path(test_data_dir) / "test/masks"
    pred_masks_folder = Path(config["result_pth_path"]).parent / "generalization" / test_dataset_name / "pred_masks"

    true_masks_files = sorted(list(Path(test_mask_folder).iterdir()))
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]

    pred_masks_files = sorted(list(pred_masks_folder.iterdir()))
    pred_masks = [read_mask_to_numpy(i) for i in pred_masks_files]

    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=0.5)
    score = ap.mean(axis=0)[0]

    return {
        "score": score,
        "ap": ap,
        "train_dataset_name": config["dataset_name"],
        "test_dataset_name": test_dataset_name,
        "train_num": config["train_num"],
        "train_id": config["train_id"],
    }


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "8"

    train_dataset = [
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
    test_dataset = [
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

    select_train_num_for_train_configs = ["1"]
    select_train_num_for_test_configs = ["1"]

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    train_configs_dict = load_configs(
        PROJECT_ROOT / "experiment_in_paper/train_image_numbers/configs",
        select_dataset=train_dataset,
        select_train_num=select_train_num_for_train_configs,
    )
    test_configs_dict = load_configs(
        PROJECT_ROOT / "experiment_in_paper/train_image_numbers/configs",
        select_dataset=test_dataset,
        select_train_num=select_train_num_for_test_configs,
    )

    dataset_to_data_dir = {}
    for config in test_configs_dict.values():
        dataset_to_data_dir[config["dataset_name"]] = config["data_dir"]

    configs = list(train_configs_dict.values())

    param_list = []
    for config in configs:
        for test_dataset_name in test_dataset:
            test_data_dir = dataset_to_data_dir[test_dataset_name]
            param_list.append(
                {"config": config, "test_dataset_name": test_dataset_name, "test_data_dir": test_data_dir}
            )

    search_space = {
        "param": tune.grid_search(param_list),
    }

    tuner = tune.Tuner(
        trainable=tune.with_resources(objective, resources={"cpu": 1, "gpu": 0}),
        param_space=search_space,
    )

    results = tuner.fit()

    df = results.get_dataframe()
    df = df[["score", "train_dataset_name", "test_dataset_name", "train_num", "train_id"]]
    df.rename(
        columns={
            "score": "ap_0.5",
            "train_dataset_name": "train_dataset_name",
            "test_dataset_name": "test_dataset_name",
            "train_num": "train_num",
            "train_id": "train_id",
        },
        inplace=True,
    )

    df = df[["train_dataset_name", "test_dataset_name", "train_num", "train_id", "ap_0.5"]]
    df.to_csv(PROJECT_ROOT / f"experiment_in_paper/result/result_TSN_generalization_{time_str}.csv", index=False)
