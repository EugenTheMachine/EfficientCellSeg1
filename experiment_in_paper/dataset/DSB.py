import shutil
import zipfile

if __name__ == "__main__":
    from project_root import DATA_ROOT

    zip_file = DATA_ROOT / "dsb2018.zip"

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(zip_file.parent)

    shutil.move(DATA_ROOT / "dsb2018", DATA_ROOT / "dsb2018_stardist")
    (DATA_ROOT / "dsb2018_stardist" / "README.md").unlink()
