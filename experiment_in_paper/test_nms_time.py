import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metrics import average_precision
from predict import load_model_from_config, sam_output_to_mask
from seg_any import (
    SamAutomaticMaskGenerator,
    SamAutomaticMaskGeneratorMaskNMS,
    SamAutomaticMaskGeneratorOptMaskNMS,
)
from set_environment import set_env


def predict_images(config, images, generator_func=None, progress_callback=None, stop_event=None):
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    model = load_model_from_config(config, empty_lora=False)
    model.eval()
    if hasattr(model, "sam"):
        model_sam = model.sam
    else:
        model_sam = model

    mask_generator = generator_func(
        model=model_sam,
        points_per_side=config["points_per_side"],
        points_per_batch=config["points_per_batch"],
        crop_n_layers=config["crop_n_layers"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        box_nms_thresh=0.0,
        crop_nms_thresh=0.0,
        pred_iou_thresh=config["pred_iou_thresh"],
        min_mask_region_area=config["min_mask_region_area"],
        max_mask_region_area_ratio=config["max_mask_region_area_ratio"],
        stability_score_thresh=config["stability_score_thresh"],
        stability_score_offset=config["stability_score_offset"],
    )

    pred_masks = []
    with torch.no_grad():
        for i, image in enumerate(tqdm(images, disable=progress_callback is not None)):
            if stop_event and stop_event.is_set():
                break
            times = []
            for t in range(11):
                start_time = time.time()
                output = mask_generator.generate(image)
                end_time = time.time()
                times.append(end_time - start_time)
                # print(f"Time taken for image {i}, {t}: {end_time - start_time} seconds")
            # print(f"Average time taken for image {i}: {np.mean(times[1:])} seconds")
            if output == []:
                mask = np.zeros_like(image[:, :, 0], dtype=np.uint16)
            else:
                mask = sam_output_to_mask(output)
            pred_masks.append(mask)

            if progress_callback:
                progress = i
                progress_callback(progress)

    return pred_masks, times[1:]


if __name__ == "__main__":
    import os

    import cv2
    import pandas as pd
    import yaml

    from data.utils import read_image_to_numpy, read_mask_to_numpy, resize_image, resize_mask
    from project_root import PROJECT_ROOT

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_path = PROJECT_ROOT / "experiment_in_paper/robustness/configs/cellseg1_cellpose_specialized_12.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    image_path = Path(config["data_dir"]) / "test/images"
    image_files = sorted(list(Path(image_path).iterdir()))[0:1]
    images = [read_image_to_numpy(i) for i in image_files]
    images = [resize_image(i, config["resize_size"]) for i in images]

    true_masks_path = Path(config["data_dir"]) / "test/masks"
    true_masks_files = sorted(list(Path(true_masks_path).iterdir()))[0:1]
    true_masks = [read_mask_to_numpy(i) for i in true_masks_files]
    true_masks = [resize_mask(i, config["resize_size"]) for i in true_masks]
    generator_func = {
        "opt_mask_nms": SamAutomaticMaskGeneratorOptMaskNMS,
        "box_nms": SamAutomaticMaskGenerator,
        "mask_nms": SamAutomaticMaskGeneratorMaskNMS,
    }
    all_times = {}
    for k, v in generator_func.items():
        pred_masks, times = predict_images(config, images, generator_func=v, progress_callback=None, stop_event=None)
        all_times[k] = times
        ap, _, _, _ = average_precision(true_masks, pred_masks, threshold=0.5)

        i = 0
        image = images[i]
        true_mask = true_masks[i]
        pred_mask = pred_masks[i]
        print(ap.mean(axis=0))

        image = image[:, :, ::-1]
        cv2.imwrite(str(PROJECT_ROOT / f"figures/images/extended_figure_3_4/pred_mask_{k}.png"), pred_mask)
    cv2.imwrite(str(PROJECT_ROOT / "figures/images/extended_figure_3_4/image.png"), image)
    time_df = pd.DataFrame(all_times)
    time_df.to_csv(PROJECT_ROOT / "figures/data/extended_figure_3_4.csv", index=False)
