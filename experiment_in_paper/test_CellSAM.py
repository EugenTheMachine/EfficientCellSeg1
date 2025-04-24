import os
import time

import numpy as np
import pandas as pd
import tqdm
import yaml
from cellSAM import segment_cellular_image

from data.dataset import TestDataset
from metrics import average_precision
from project_root import DATA_ROOT, PROJECT_ROOT

# you need install cellSAM to use this script
# https://github.com/vanvalenlab/cellSAM

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    available_test_datasets = [
        "cellpose_generalized",
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
    with open(
        PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml", "r"
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    results = {}
    for i in range(len(available_test_datasets)):
        selected_test_dataset = available_test_datasets[i]
        resize_size = config["dataset"][selected_test_dataset]["resize_size"]
        test_dataset = TestDataset(
            image_dir=DATA_ROOT / f"{selected_test_dataset}/test/images",
            mask_dir=DATA_ROOT / f"{selected_test_dataset}/test/masks",
            resize_size=resize_size,
        )
        pred_masks = []
        true_masks = []
        images = []
        for i, (image, true_mask) in enumerate(tqdm.tqdm(test_dataset)):
            if selected_test_dataset == "cellpose_specialized":
                image = image[:, :, [2, 0, 1]]
            elif selected_test_dataset == "cellpose_generalized":
                image = image[:, :, [2, 0, 1]]
                # chech if red channel is all 0
                if not (image[:, :, 0] == 0).all():
                    image[:, :, 1] = 0
            elif selected_test_dataset == "cellseg_blood":
                image = image.mean(axis=-1).astype(np.uint8)
                image = np.stack([image, image, image], axis=-1)
            elif selected_test_dataset == "dsb2018_stardist":
                image[:, :, 1] = 0
            elif selected_test_dataset in [
                "deepbacs_rod_brightfield",
                "deepbacs_rod_fluorescence",
            ]:
                image[:, :, 1] = 0
            elif selected_test_dataset.startswith("tissuenet"):
                image = image[:, :, [2, 0, 1]]
                image[:, :, 1] = 0
            # set red channel to 0
            image[:, :, 0] = 0
            images.append(image)
            pred_mask, _, _ = segment_cellular_image(image / 255.0, device="cuda")
            pred_masks.append(pred_mask)
            true_masks.append(true_mask)

        ap, _, _, _ = average_precision(true_masks, pred_masks, threshold=0.5)
        mean_ap = ap.mean(axis=0)[0]
        results[selected_test_dataset] = mean_ap
        print(f"{selected_test_dataset}: {mean_ap:.8f}")
    # save result
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    df = pd.DataFrame(data=[results]).T
    print(df)
    df.to_csv(
        PROJECT_ROOT / f"experiment_in_paper/result/result_CellSAM_{time_str}.csv"
    )
