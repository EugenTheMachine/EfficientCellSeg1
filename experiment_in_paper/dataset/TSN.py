import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from rich.progress import track
from skimage.exposure import rescale_intensity

from data.utils import remap_mask_color


def convert_to_rgb_image(data, colors):
    valid = ["red", "green", "blue"]
    colors = [c.lower() for c in colors]

    rgb = np.zeros(data.shape[:3] + (3,), dtype="float32")

    for i in range(data.shape[0]):
        for c in range(data.shape[-1]):
            img = data[i, :, :, c]
            nonzero = img[np.nonzero(img)]
            if len(nonzero) > 0:
                p5, p95 = np.percentile(nonzero, [5, 95])
                scaled = rescale_intensity(img, in_range=(p5, p95), out_range="float32")
                idx = np.where(np.isin(valid, colors[c]))
                rgb[i, :, :, idx] = scaled
    return rgb


def load_npz(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = np.load(npz_file, allow_pickle=True)
    df = pd.DataFrame(data["meta"])

    new_header = df.iloc[0]
    df = df[3:] if npz_file.stem == "tissuenet_v1.1_test" else df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    print(f"Loading images and masks from {npz_file}")
    two_channel_images, masks = data["X"], data["y"]
    rgb_images = (255 * convert_to_rgb_image(two_channel_images, colors=["green", "blue"])).astype(np.uint8)

    masks = masks.astype(np.uint16)
    for i in track(range(masks.shape[0])):
        for j in range(masks.shape[3]):
            masks[i, :, :, j] = remap_mask_color(masks[i, :, :, j])

    return rgb_images, masks, df


def count_df(df: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame(
        {
            "folder": df["experiment"].apply(lambda x: str(Path(x).parent.stem).split("-")[-1]),
            "tissue": df["experiment"].apply(lambda x: str(Path(x).stem)),
            "specimen": df["specimen"],
        }
    )
    return new_df.groupby(["folder", "tissue", "specimen"]).size().reset_index(name="count")


def write_images_to_folder(images: np.ndarray, masks: np.ndarray, folder: Path):
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)
    images_dir = folder / "images"
    masks_dir = folder / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    for i in range(images.shape[0]):
        np.save(images_dir / f"{i}.npy", images[i])
        np.save(masks_dir / f"{i}.npy", masks[i])


def generate_dataset(
    root_dir: Path,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    folder: str,
    tissue: str,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    test_images: np.ndarray,
    test_masks: np.ndarray,
):
    train_index = get_selected_index(train_meta, folder=folder, tissue=tissue)
    test_index = get_selected_index(test_meta, folder=folder, tissue=tissue)

    nuclei_folder = root_dir / f"tissuenet_{folder}_{tissue}_nuclei"
    write_images_to_folder(train_images[train_index], train_masks[train_index, :, :, 1], nuclei_folder / "train")
    write_images_to_folder(test_images[test_index], test_masks[test_index, :, :, 1], nuclei_folder / "test")


def get_selected_index(df: pd.DataFrame, folder: str = "", tissue: str = "") -> np.ndarray:
    if folder:
        df = df[df["experiment"].apply(lambda x: str(Path(x).parent.stem).split("-")[-1]) == folder]
    if tissue:
        df = df[df["experiment"].apply(lambda x: str(Path(x).stem)) == tissue]
    return df.index.values


def load_or_create_all_data(root_dir: Path) -> Dict[str, Any]:
    all_data_file = root_dir / "tissuenet_v1.1.pkl"

    if all_data_file.exists():
        with open(all_data_file, "rb") as f:
            return pickle.load(f)

    data = {}
    for split in ["train", "val", "test"]:
        file = root_dir / f"tissuenet_v1.1_{split}.npz"
        images, masks, meta = load_npz(file)
        data[f"{split}_images"] = images
        data[f"{split}_masks"] = masks
        data[f"{split}_meta"] = meta

    with open(all_data_file, "wb") as f:
        pickle.dump(data, f)

    return data


if __name__ == "__main__":
    import zipfile

    from project_root import DATA_ROOT

    root_dir = DATA_ROOT
    zip_file = root_dir / "tissuenet_v1.1.zip"
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(zip_file.parent)

    all_data = load_or_create_all_data(root_dir)

    count_train = count_df(all_data["train_meta"])

    for _, row in count_train[count_train["count"] >= 10].iterrows():
        folder, tissue, count = row["folder"], row["tissue"], row["count"]
        print(f"Processing: {folder} {tissue} {count}")
        generate_dataset(
            root_dir,
            all_data["train_meta"],
            all_data["test_meta"],
            folder,
            tissue,
            all_data["train_images"],
            all_data["train_masks"],
            all_data["test_images"],
            all_data["test_masks"],
        )

    (root_dir / "tissuenet_v1.1.pkl").unlink()
    (root_dir / "tissuenet_v1.1_test.npz").unlink()
    (root_dir / "tissuenet_v1.1_train.npz").unlink()
    (root_dir / "tissuenet_v1.1_val.npz").unlink()
    (root_dir / "README.md").unlink()
    (root_dir / "metadata_v1.1.yaml").unlink()
    (root_dir / "LICENSE").unlink()
