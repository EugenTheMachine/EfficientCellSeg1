import shutil
import zipfile
from pathlib import Path
from project_root import DATA_ROOT
# DATA_ROOT = Path("./data/cellseg1")

if __name__ == "__main__":
    train_zip_file = DATA_ROOT / "train.zip"
    test_zip_file = DATA_ROOT / "test.zip"

    with zipfile.ZipFile(train_zip_file, "r") as zip_ref:
        zip_ref.extractall(train_zip_file.parent)
    with zipfile.ZipFile(test_zip_file, "r") as zip_ref:
        zip_ref.extractall(test_zip_file.parent)

    root_dir = DATA_ROOT / "cellpose_generalized"
    root_dir.mkdir(exist_ok=True)
    shutil.move(train_zip_file.parent / "train", root_dir / "train")
    shutil.move(train_zip_file.parent / "test", root_dir / "test")

    (root_dir / "train" / "images").mkdir(exist_ok=True)
    (root_dir / "train" / "masks").mkdir(exist_ok=True)
    train_images = list(Path(root_dir / "train").glob("*_img.png"))
    train_masks = list(Path(root_dir / "train").glob("*_masks.png"))
    for img, mask in zip(train_images, train_masks):
        shutil.move(img, root_dir / "train" / "images" / img.name)
        shutil.move(mask, root_dir / "train" / "masks" / mask.name)

    (root_dir / "test" / "images").mkdir(exist_ok=True)
    (root_dir / "test" / "masks").mkdir(exist_ok=True)
    test_images = list(Path(root_dir / "test").glob("*_img.png"))
    test_masks = list(Path(root_dir / "test").glob("*_masks.png"))
    for img, mask in zip(test_images, test_masks):
        shutil.move(img, root_dir / "test" / "images" / img.name)
        shutil.move(mask, root_dir / "test" / "masks" / mask.name)
