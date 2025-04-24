import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import sys
sys.path.append(".")

from data.utils import calculate_cell_statistics, read_mask_to_numpy


def generate_data_statistics(data_root_dir, save_cell_statistics=True):
    data_dirs = sorted(list(data_root_dir.iterdir()))
    all_stats = []

    valid_image_extensions = [
        ".tif",
        ".tiff",
        ".TIF",
        ".TIFF",
        ".bmp",
        ".png",
        ".BMP",
        ".PNG",
        ".npy",
        ".nii",
        ".nii.gz",
    ]
    valid_dataset = [
        "livecell",
        # "cellpose_generalized",
        # "cellpose_specialized",
        # "cellseg_blood",
        # "dsb2018_stardist",
        # "deepbacs_rod_brightfield",
        # "deepbacs_rod_fluorescence",
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
    dataset_dict = {}
    for data_dir in data_dirs:
        dataset_name = data_dir.stem
        print(f"dataset_name = {dataset_name}")
        if dataset_name not in valid_dataset:
            continue

        if dataset_name in [
            "dsb2018_stardist",
            "cellpose_specialized",
            "cellpose_generalized",
        ]:
            resize_size = [512, 512]
        elif dataset_name.startswith("tissuenet_"):
            resize_size = None
        elif dataset_name == "cellseg_blood":
            resize_size = [512, 512]
        elif dataset_name.startswith("deepbacs_"):
            resize_size = [512, 512]
        else:
            resize_size = [512, 512]

        train_image_files = []
        test_image_files = []
        train_mask_files = []
        test_mask_files = []
        for extension in valid_image_extensions:
            train_image_files.extend(list(data_dir.glob(f"train/masks/*{extension}")))
            test_image_files.extend(list(data_dir.glob(f"test/masks/*{extension}")))
            train_mask_files.extend(list(data_dir.glob(f"train/masks/*{extension}")))
            test_mask_files.extend(list(data_dir.glob(f"test/masks/*{extension}")))
        train_image_files = sorted(train_image_files)
        test_image_files = sorted(test_image_files)
        train_mask_files = sorted(train_mask_files)
        test_mask_files = sorted(test_mask_files)
        assert len(train_image_files) == len(train_mask_files)
        assert len(test_image_files) == len(test_mask_files)
        train_image_num = len(train_mask_files)
        test_image_num = len(test_mask_files)

        if save_cell_statistics:
            train_stats = []
            test_stats = []
            total_train_cells = 0
            total_test_cells = 0
            all_train_cell_sizes = []
            all_test_cell_sizes = []

            print(f"Processing {dataset_name} training masks...")
            for idx, mask_file in enumerate(tqdm(train_mask_files)):
                mask = read_mask_to_numpy(mask_file)
                cell_number, cell_size = calculate_cell_statistics(mask)
                total_train_cells += cell_number
                all_train_cell_sizes.extend(cell_size)
                stats_dict = {
                    "dataset_name": dataset_name,
                    "split": "train",
                    "file_name": mask_file.name,
                    "index": idx,
                    "cell_number": cell_number,
                    "cell_sizes": cell_size.tolist() if len(cell_size) > 0 else [],
                    "mean_cell_size": float(np.mean(cell_size)) if len(cell_size) > 0 else 0,
                }
                train_stats.append(stats_dict)
                all_stats.append(stats_dict)

            print(f"Processing {dataset_name} test masks...")
            for idx, mask_file in enumerate(tqdm(test_mask_files)):
                mask = read_mask_to_numpy(mask_file)
                cell_number, cell_size = calculate_cell_statistics(mask)
                total_test_cells += cell_number
                all_test_cell_sizes.extend(cell_size)
                stats_dict = {
                    "dataset_name": dataset_name,
                    "split": "test",
                    "file_name": mask_file.name,
                    "index": idx,
                    "cell_number": cell_number,
                    "cell_sizes": cell_size.tolist() if len(cell_size) > 0 else [],
                    "mean_cell_size": float(np.mean(cell_size)) if len(cell_size) > 0 else 0,
                }
                test_stats.append(stats_dict)
                all_stats.append(stats_dict)

            dataset_dict[dataset_name] = {
                "data_dir": str(data_dir),
                "train_image_num": train_image_num,
                "test_image_num": test_image_num,
                "resize_size": resize_size,
                "train_statistics": {
                    "total_cells": total_train_cells,
                    "mean_cell_size": float(np.mean(all_train_cell_sizes)),
                },
                "test_statistics": {
                    "total_cells": total_test_cells,
                    "mean_cell_size": float(np.mean(all_test_cell_sizes)),
                },
            }
        else:
            dataset_dict[dataset_name] = {
                "data_dir": str(data_dir),
                "train_image_num": train_image_num,
                "test_image_num": test_image_num,
                "resize_size": resize_size,
            }
    return dataset_dict, all_stats


if __name__ == "__main__":
    from project_root import DATA_ROOT, PROJECT_ROOT

    data_root_dir = DATA_ROOT
    save_file = PROJECT_ROOT / "experiment_in_paper/dataset/data_statistics.yaml"
    csv_save_file = PROJECT_ROOT / "experiment_in_paper/dataset/per_image_statistics.csv"

    data_statistics, all_stats = generate_data_statistics(data_root_dir)

    with open(save_file, "w") as f:
        yaml.dump({"dataset": data_statistics}, f, default_flow_style=True)

    all_stats_df = pd.DataFrame(all_stats)
    all_stats_df.to_csv(csv_save_file, index=False)
