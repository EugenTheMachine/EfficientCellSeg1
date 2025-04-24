import numpy as np

from data.utils import read_image_to_numpy, read_mask_to_numpy
from project_root import DATA_ROOT


def load_gt_images(dataset, select_ids=None, train_or_test="train"):
    base_dir = DATA_ROOT
    image_path = base_dir / f"{dataset}/{train_or_test}/images"
    mask_path = base_dir / f"{dataset}/{train_or_test}/masks"

    all_image_paths = sorted(list(image_path.glob("*")))
    all_mask_paths = sorted(list(mask_path.glob("*")))

    if select_ids is not None:
        if isinstance(select_ids, int):
            select_ids = [select_ids]
        image_paths = [all_image_paths[i] for i in select_ids]
        mask_paths = [all_mask_paths[i] for i in select_ids]
    else:
        image_paths = all_image_paths
        mask_paths = all_mask_paths

    images = [read_image_to_numpy(str(image_path)) for image_path in image_paths]  # noqa: F841
    masks = [read_mask_to_numpy(str(mask_path)) for mask_path in mask_paths]

    num_cells = []
    for mask in masks:
        unique_labels = np.unique(mask)
        num_cell = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        num_cells.append(num_cell)

    return images, masks, num_cells


def load_pred_masks(dataset, train_id=None, test_ids=None):
    pred_mask_folder = DATA_ROOT / f"{dataset}/cellseg1/robustness/cellseg1_{dataset}_{train_id}/pred_masks"
    all_mask_paths = sorted(list(pred_mask_folder.glob("*")))
    mask_paths = [all_mask_paths[i] for i in test_ids]
    masks = [read_mask_to_numpy(str(mask_path)) for mask_path in mask_paths]
    return masks


def load_multiple_gt_images(dataset_ids, train_or_test="train"):
    results = {}

    for dataset, select_ids in dataset_ids.items():
        images, masks, num_cells = load_gt_images(dataset, select_ids, train_or_test)
        results[dataset] = {"images": images, "masks": masks, "num_cells": num_cells}

    return results
