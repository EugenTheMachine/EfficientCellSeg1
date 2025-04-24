import shutil
import zipfile
from pathlib import Path

import numpy as np

from data.utils import read_mask_to_numpy


def convert_mask(mask_folder):
    mask_files = sorted(list(Path(mask_folder).iterdir()))
    for file in mask_files:
        mask = read_mask_to_numpy(file)
        file.unlink()
        save_path = file.with_suffix(".npy")
        np.save(str(save_path), mask)


if __name__ == "__main__":
    from project_root import DATA_ROOT

    zip_file = DATA_ROOT / "DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip"

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(zip_file.parent)

    shutil.rmtree(DATA_ROOT / "live-cell_data")

    shutil.move(DATA_ROOT / "train", DATA_ROOT / "deepbacs_rod_brightfield" / "train")
    (DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "brightfield").rename(
        DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "images"
    )
    (DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "masks_RoiMap").rename(
        DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "masks"
    )
    shutil.rmtree(DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "masks_binary")
    convert_mask(DATA_ROOT / "deepbacs_rod_brightfield" / "train" / "masks")

    shutil.move(DATA_ROOT / "test", DATA_ROOT / "deepbacs_rod_brightfield" / "test")
    (DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "brightfield").rename(
        DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "images"
    )
    (DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "masks_RoiMap").rename(
        DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "masks"
    )
    shutil.rmtree(DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "masks_binary")
    convert_mask(DATA_ROOT / "deepbacs_rod_brightfield" / "test" / "masks")
