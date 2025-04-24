import shutil
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.utils import read_mask_to_numpy


def convert_mask(mask_folder):
    mask_files = sorted(list(Path(mask_folder).iterdir()))
    for file in tqdm(mask_files):
        mask = read_mask_to_numpy(file)
        file.unlink()
        save_path = file.with_suffix(".npy")
        np.save(str(save_path), mask)


if __name__ == "__main__":
    from project_root import DATA_ROOT

    zip_file = DATA_ROOT / "DeepBacs_Data_Segmentation_B.subtilis_FtsZ_dataset.zip"

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(zip_file.parent)

    shutil.rmtree(DATA_ROOT / "CARE_U-Net_dataset")
    shutil.rmtree(DATA_ROOT / "SplineDist_dataset")
    shutil.rmtree(DATA_ROOT / "StarDist_dataset")
    (DATA_ROOT / "pix2pix_dataset" / "Notes_pix2pix.txt").unlink()

    (DATA_ROOT / "pix2pix_dataset").rename(DATA_ROOT / "deepbacs_rod_fluorescence")
    (DATA_ROOT / "deepbacs_rod_fluorescence/train/fluorescence/").rename(
        DATA_ROOT / "deepbacs_rod_fluorescence/train/images/"
    )
    (DATA_ROOT / "deepbacs_rod_fluorescence/test/fluorescence/").rename(
        DATA_ROOT / "deepbacs_rod_fluorescence/test/images/"
    )
    convert_mask(DATA_ROOT / "deepbacs_rod_fluorescence/train/masks/")
    convert_mask(DATA_ROOT / "deepbacs_rod_fluorescence/test/masks/")
