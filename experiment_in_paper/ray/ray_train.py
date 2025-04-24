import os
import logging
import time

from ray import tune

from cellseg1_train import main
from experiment_in_paper.ray.utils import load_configs
from project_root import PROJECT_ROOT


def objective(config):
    config = config["config"]
    start_time = time.time()
    _ = main(config)
    end_time = time.time()
    return {"config": config, "score": (end_time - start_time) * 1000}


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # having only 1 CUDA device
    # os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "8"  # NOTE: maybe needs to be reduced
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "4"
    select_dataset = [
        "livecell"
        # "cellpose_generalized",
        # "cellpose_specialized",
        # "cellseg_blood",
        # "deepbacs_rod_brightfield",
        # "deepbacs_rod_fluorescence",
        # "dsb2018_stardist",
        # "tissuenet_Breast_20191211_IMC_nuclei",
        # "tissuenet_Breast_20200116_DCIS_nuclei",
        # "tissuenet_Breast_20200526_COH_BC_nuclei",
        # "tissuenet_Epidermis_20200226_Melanoma_nuclei",
        # "tissuenet_Epidermis_20200623_sizun_epidermis_nuclei",
        # "tissuenet_GI_20191219_Eliot_nuclei",
        # "tissuenet_GI_20200219_Roshan_nuclei",
        # "tissuenet_GI_20200627_CODEX_CRC_nuclei",
        # "tissuenet_Lung_20200210_CyCIF_Lung_LN_nuclei",
        # "tissuenet_Lymph_Node_20200114_cHL_nuclei",
        # "tissuenet_Lymph_Node_20200520_HIV_nuclei",
        # "tissuenet_Pancreas_20200512_Travis_PDAC_nuclei",
        # "tissuenet_Pancreas_20200624_CODEX_Panc_nuclei",
        # "tissuenet_Tonsil_20200211_CyCIF_Tonsil_nuclei",
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
    # print(search_space)

    # tuner = tune.Tuner(
    #     trainable=tune.with_resources(objective, resources={"cpu": 5, "gpu": 1}),
    #     param_space=search_space,
    # )
    tuner = tune.Tuner(
        trainable=tune.with_resources(objective, resources={"cpu": 1, "gpu": 1}),
        param_space=search_space,
    )
    # tune.logger.setLevel(logging.ERROR)
    results = tuner.fit(verbose=0)

    (PROJECT_ROOT / "experiment_in_paper/result/").mkdir(exist_ok=True)

    df = results.get_dataframe()
    df.to_csv(
        PROJECT_ROOT / f"experiment_in_paper/result/train_{time_str}.csv", index=False
    )
