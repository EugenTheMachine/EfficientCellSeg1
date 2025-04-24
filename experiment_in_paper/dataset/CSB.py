import shutil
import zipfile

if __name__ == "__main__":
    from project_root import DATA_ROOT

    train_zip_file = DATA_ROOT / "Training-labeled.zip"
    test_zip_file = DATA_ROOT / "Tuning.zip"

    with zipfile.ZipFile(train_zip_file, "r") as zip_ref:
        zip_ref.extractall(train_zip_file.parent)
    with zipfile.ZipFile(test_zip_file, "r") as zip_ref:
        zip_ref.extractall(test_zip_file.parent)

    root_dir = DATA_ROOT / "cellseg_blood"
    root_dir.mkdir(exist_ok=True)

    shutil.move(DATA_ROOT / "Training-labeled", root_dir / "train")
    shutil.move(root_dir / "train" / "labels", root_dir / "train" / "masks")
    shutil.move(DATA_ROOT / "Tuning", root_dir / "test")
    shutil.move(root_dir / "test" / "labels", root_dir / "test" / "masks")

    train_image_files = sorted(list((root_dir / "train" / "images").glob("*")))
    train_masks_files = sorted(list((root_dir / "train" / "masks").glob("*")))

    test_image_files = sorted(list((root_dir / "test" / "images").glob("*")))
    test_masks_files = sorted(list((root_dir / "test" / "masks").glob("*")))

    blood_train_ids = list(range(0, 12)) + list(range(14, 141))
    blood_test_ids = [0, 2, 6, 7, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23]

    for i in range(len(train_image_files)):
        if i not in blood_train_ids:
            train_image_files[i].unlink()
            train_masks_files[i].unlink()

    for i in range(len(test_image_files)):
        if i not in blood_test_ids:
            test_image_files[i].unlink()
            test_masks_files[i].unlink()
