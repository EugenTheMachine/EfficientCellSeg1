import os
import time

import numpy as np
import yaml
from ray import tune
from segment_anything_org.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything_org.build_sam import build_sam_vit_h

from data.dataset import TestDataset
from metrics import average_precision
from project_root import DATA_ROOT, PROJECT_ROOT
from set_environment import set_env

# you need download original sam code and rename it to segment_anything_org to use this script
# https://github.com/facebookresearch/segment-anything/tree/main/segment_anything


def remap_mask_color(mask, continual=True, random=False):
    mask = mask.astype(np.uint16)
    color_with_background_0 = sorted(np.unique(mask))
    if len(color_with_background_0) == 1:
        return mask
    color = color_with_background_0[1:]
    if (color[0] == 1) and (len(color) == max(color)) and (not random):
        return mask
    color_ori = color.copy()
    if continual:
        color = [i + 1 for i in range(len(color))]
    if random:
        np.random.shuffle(color)
    new_true = np.zeros_like(mask)
    for i, c in enumerate(color_ori):
        new_true[mask == c] = color[i]
    return new_true


def merge_everything_model_output(sam_output):
    if len(sam_output) == 0:
        return None
    shape = sam_output[0]["segmentation"].shape
    pred_instance_mask = np.zeros(shape, dtype=np.int64)
    for i, o in enumerate(sam_output):
        pred_instance_mask[o["segmentation"]] = 0
        pred_instance_mask += (i + 1) * o["segmentation"]
    pred_instance_mask = remap_mask_color(pred_instance_mask)
    return pred_instance_mask


def objective(config):
    set_env()
    dataset_name = config["dataset"]
    checkpoint = PROJECT_ROOT / "streamlit_storage/sam_backbone/sam_vit_h_4b8939.pth"
    sam = build_sam_vit_h(str(checkpoint)).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    with open(
        PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml", "r"
    ) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    resize_size = data_config["dataset"][dataset_name]["resize_size"]
    test_dataset = TestDataset(
        image_dir=DATA_ROOT / dataset_name / "test" / "images",
        mask_dir=DATA_ROOT / dataset_name / "test" / "masks",
        resize_size=resize_size,
    )

    true_masks = []
    pred_masks = []

    for image, mask in test_dataset:
        sam_output = mask_generator.generate(image)
        pred_mask = merge_everything_model_output(sam_output)
        if pred_mask is None:
            pred_mask = np.zeros_like(mask, dtype=np.uint16)
        true_masks.append(mask)
        pred_masks.append(pred_mask)

    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=[0.5])
    mean_ap = ap.mean(axis=0)[0]

    return {"score": mean_ap, "mean_ap": mean_ap}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "8"

    selected_test_datasets = [
        "cellpose_specialized",
        "cellseg_blood",
        "dsb2018_stardist",
        "deepbacs_rod_brightfield",
        "deepbacs_rod_fluorescence",
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

    search_space = {
        "dataset": tune.grid_search(selected_test_datasets),
    }

    tuner = tune.Tuner(
        trainable=tune.with_resources(objective, resources={"cpu": 1, "gpu": 1}),
        param_space=search_space,
    )

    results = tuner.fit()

    # Print results
    for result in results:
        print(
            f"Dataset: {result.config['dataset']}, Mean AP: {result.metrics['mean_ap']}"
        )
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    df = results.get_dataframe()
    # select column "config/dataset", "score"
    df = df[["config/dataset", "score"]]
    df.rename(columns={"config/dataset": "dataset", "score": "mean_ap"}, inplace=True)
    df.to_csv(
        PROJECT_ROOT / f"experiment_in_paper/result/result_SAM_{time_str}.csv",
        index=False,
    )
