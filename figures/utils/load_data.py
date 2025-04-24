import json

import pandas as pd

from data.utils import resize_image, resize_mask
from figures.utils.load_image import load_multiple_gt_images
from project_root import PROJECT_ROOT


def load_color():
    colors = {
        "cellseg1": "#E14774",
        "cellpose-cyto": "#1C8CA9",
        "cellpose-cyto2": "#7058BD",
        "cellpose-scratch": "#CC7A0B",
        "stardist": "#269D68",
        "SAM": "#E16031",
        "CellSAM": "#935355",
    }
    return colors


def load_short_names():
    short_name = {
        "cellpose_specialized": "CPS",
        "cellpose_generalized": "CPG",
        "cellseg_blood": "CSB",
        "dsb2018_stardist": "DSB",
        "deepbacs_rod_brightfield": "ECB",
        "deepbacs_rod_fluorescence": "BSF",
        "tissuenet_Breast_20191211_IMC_nuclei": "TSN-1",
        "tissuenet_Breast_20200116_DCIS_nuclei": "TSN-2",
        "tissuenet_Breast_20200526_COH_BC_nuclei": "TSN-3",
        "tissuenet_Epidermis_20200226_Melanoma_nuclei": "TSN-4",
        "tissuenet_Epidermis_20200623_sizun_epidermis_nuclei": "TSN-5",
        "tissuenet_GI_20191219_Eliot_nuclei": "TSN-6",
        "tissuenet_GI_20200219_Roshan_nuclei": "TSN-7",
        "tissuenet_GI_20200627_CODEX_CRC_nuclei": "TSN-8",
        "tissuenet_Lung_20200210_CyCIF_Lung_LN_nuclei": "TSN-9",
        "tissuenet_Lymph_Node_20200114_cHL_nuclei": "TSN-10",
        "tissuenet_Lymph_Node_20200520_HIV_nuclei": "TSN-11",
        "tissuenet_Pancreas_20200512_Travis_PDAC_nuclei": "TSN-12",
        "tissuenet_Pancreas_20200624_CODEX_Panc_nuclei": "TSN-13",
        "tissuenet_Tonsil_20200211_CyCIF_Tonsil_nuclei": "TSN-14",
    }
    reversed_short_name = {v: k for k, v in short_name.items()}
    return short_name, reversed_short_name


def load_figure_1_1_data():
    with open(PROJECT_ROOT / "figures/data/figure_1_1.json", "r") as f:
        dicts, max_id_dict = json.load(f)
    return dicts, max_id_dict


def load_figure_1_2_data():
    with open(PROJECT_ROOT / "figures/data/figure_1_2.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_3_data():
    df = pd.read_csv(PROJECT_ROOT / "figures/data/figure_3.csv")
    return df


def load_figure_3_1_data():
    with open(PROJECT_ROOT / "figures/data/figure_3_1.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_3_2_data():
    with open(PROJECT_ROOT / "figures/data/figure_3_2.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_3_3_data():
    with open(PROJECT_ROOT / "figures/data/figure_3_3.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_4_1_data():
    with open(PROJECT_ROOT / "figures/data/figure_4_1.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_4_2_data():
    with open(PROJECT_ROOT / "figures/data/figure_4_2.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_5_1_data():
    with open(PROJECT_ROOT / "figures/data/figure_5_1.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_5_2_data():
    with open(PROJECT_ROOT / "figures/data/figure_5_2.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_figure_5_3_data():
    cellseg1_df = pd.read_csv(
        PROJECT_ROOT / "figures/data/figure_5_3_cellseg1.csv", index_col=0
    )
    cellpose_cyto2_df = pd.read_csv(
        PROJECT_ROOT / "figures/data/figure_5_3_cellpose-cyto2.csv", index_col=0
    )
    stardist_df = pd.read_csv(
        PROJECT_ROOT / "figures/data/figure_5_3_stardist.csv", index_col=0
    )
    result = {
        "cellseg1": cellseg1_df,
        "cellpose-cyto2": cellpose_cyto2_df,
        "stardist": stardist_df,
    }
    return result


def load_ap_50_for_train_ids(image_ids_for_all_datasets, method):
    df = pd.read_csv(PROJECT_ROOT / "figures/data/figure_3.csv")
    ap_50_for_train_ids = {}
    for dataset in image_ids_for_all_datasets.keys():
        ap_50_for_train_ids[dataset] = {}
        for train_id in image_ids_for_all_datasets[dataset]:
            row = df[
                (df["train_id"] == train_id)
                & (df["method_name"] == method)
                & (df["dataset_name"] == dataset)
            ]
            ap_50_for_train_ids[dataset][train_id] = row["ap_0.5"].values[0]
    return ap_50_for_train_ids


def load_extended_figure_2_data():
    image_ids_for_all_datasets = {
        "cellseg_blood": [7],
    }

    image_and_gt_data = load_multiple_gt_images(image_ids_for_all_datasets, "train")
    image = image_and_gt_data["cellseg_blood"]["images"][0]
    true_mask = image_and_gt_data["cellseg_blood"]["masks"][0]
    image_size = image.shape[:2]
    image_size = (768, 1020)
    image = resize_image(image, image_size)
    true_mask = resize_mask(true_mask, image_size)
    crop_box = (280, 320, 550, 940)
    image = image[crop_box[0] : crop_box[2], crop_box[1] : crop_box[3]]
    true_mask = true_mask[crop_box[0] : crop_box[2], crop_box[1] : crop_box[3]]
    contour_color = "#EFEF00"
    box_color = "#9ef01a"
    overlap_box_color = "#e5383b"
    return image, true_mask, contour_color, box_color, overlap_box_color


def load_extended_figure_2_1_data():
    with open(PROJECT_ROOT / "figures/data/figure_3_3.json", "r") as f:
        dicts = json.load(f)
    good_ap50s = dicts["good_ap50s"]
    train_ids_all = dicts["good_train_ids"]
    return train_ids_all, good_ap50s


def load_extended_figure_3_4_data():
    df = pd.read_csv(PROJECT_ROOT / "figures/data/extended_figure_3_4.csv")
    return df


def load_extended_figure_4_1_data():
    df = pd.read_csv(PROJECT_ROOT / "figures/data/extended_figure_4_1.csv")
    df = df.drop(columns=["train_id", "train_num"])
    return df


def load_extended_figure_4_2_data():
    with open(PROJECT_ROOT / "figures/data/extended_figure_4_2.json", "r") as f:
        dicts = json.load(f)
    return dicts


def load_extended_figure_5_data():
    df = pd.read_csv(PROJECT_ROOT / "figures/data/extended_figure_5.csv")
    short_name, _ = load_short_names()
    df["dataset_name"] = df["dataset_name"].map(short_name)
    order = ["CSB", "ECB", "BSF", "DSB", "CPS"]
    df["dataset_name"] = pd.Categorical(
        df["dataset_name"], categories=order, ordered=True
    )
    df = df.sort_values("dataset_name")
    return df


if __name__ == "__main__":
    image_ids_for_all_datasets = {
        "cellseg_blood": [1, 2, 3, 4, 5],
        "deepbacs_rod_brightfield": [1, 2, 3, 4, 5],
        "deepbacs_rod_fluorescence": [1, 2, 3, 4, 5],
        "dsb2018_stardist": [104, 111, 113, 122, 124, 128],
        "cellpose_specialized": [1, 2, 3, 4, 5],
    }
    ap_50_for_train_ids = load_ap_50_for_train_ids(
        image_ids_for_all_datasets, "cellseg1"
    )
