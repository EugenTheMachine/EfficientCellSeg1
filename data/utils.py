from pathlib import Path
from typing import List, Tuple, Union

import cv2
import nibabel as nib
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import label
from skimage import exposure, measure
from skimage.transform import resize


def calculate_cell_statistics(mask):
    mask = mask.astype(int)
    unique, counts = np.unique(mask, return_counts=True)
    if len(unique) == 1:
        return 0, []
    cell_number = len(unique) - 1
    cell_size = counts[1:]
    return cell_number, cell_size


def resize_mask(mask, resize_size):
    if (resize_size is not None) and (resize_size[0] > 0) and (resize_size[1] > 0):
        mask = cv2.resize(
            mask.astype(np.uint16),
            (resize_size[1], resize_size[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
    return mask.astype(np.uint16)


def resize_image(image, resize_size):
    if (resize_size is not None) and (resize_size[0] > 0) and (resize_size[1] > 0):
        image = cv2.resize(
            image,
            (resize_size[1], resize_size[0]),
            interpolation=cv2.INTER_LINEAR_EXACT,
        )
    return image


def masks_to_bboxes(mask):
    bboxes = []
    props = measure.regionprops(mask)
    for prop in props:
        bbox = prop.bbox
        bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
    return bboxes


def calcualte_overlap_box(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    x_min_overlap = max(x_min1, x_min2)
    y_min_overlap = max(y_min1, y_min2)
    x_max_overlap = min(x_max1, x_max2)
    y_max_overlap = min(y_max1, y_max2)
    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        return None
    else:
        return [x_min_overlap, y_min_overlap, x_max_overlap, y_max_overlap]


def find_all_valid_overlap_boxes(boxs):
    overlap_boxes = []
    for i in range(len(boxs)):
        for j in range(i + 1, len(boxs)):
            overlap_box = calcualte_overlap_box(boxs[i], boxs[j])
            if overlap_box is not None:
                overlap_boxes.append(overlap_box)
    return overlap_boxes


# this function come from cellpose
# https://github.com/MouseLand/cellpose/blob/main/cellpose/
# Copyright © 2020 Howard Hughes Medical Institute
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of HHMI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles


    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]]

    return IMG, ysub, xsub, Ly, Lx


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    instance_mask, num_instances = label(mask)
    if num_instances <= 1:
        return mask
    sizes = [np.sum(instance_mask == i) for i in range(1, num_instances + 1)]
    largest_instance = np.argmax(sizes) + 1
    return instance_mask == largest_instance


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    return exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)


def all_channels_are_equal(image: np.ndarray) -> bool:
    first_channel = image[..., 0]
    for channel_idx in range(1, image.shape[2]):
        if not np.array_equal(first_channel, image[..., channel_idx]):
            return False
    return True


def get_non_empty_channels(image: np.ndarray) -> List[int]:
    return [i for i in range(image.shape[-1]) if not np.all(image[:, :, i] == 0)]


def binary_mask_to_instance_mask(binary_mask: np.ndarray) -> np.ndarray:
    assert binary_mask.ndim == 2
    return label(binary_mask)[0]


def rgb_mask_to_uint16_mask(rgb_array: np.ndarray) -> np.ndarray:
    assert rgb_array.shape[-1] == 3
    flat_rgb_array = rgb_array.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat_rgb_array, axis=0, return_inverse=True)
    return inverse.reshape(rgb_array.shape[:2]).astype(np.uint16)


def remap_mask_color(mask: np.ndarray, continual: bool = True, random: bool = False) -> np.ndarray:
    mask = mask.astype(np.uint16)

    if mask.ndim == 3 and mask.shape[-1] == 3:
        if all_channels_are_equal(mask):
            mask = mask[:, :, 0]
        else:
            mask = rgb_mask_to_uint16_mask(mask)

    color_with_background_0 = sorted(np.unique(mask))
    if len(color_with_background_0) == 1:
        return mask

    color = color_with_background_0[1:]
    if color[0] == 1 and len(color) == max(color) and not random:
        return mask

    color_ori = color.copy()
    if continual:
        color = list(range(1, len(color) + 1))
    if random:
        np.random.shuffle(color)

    new_true = np.zeros_like(mask)
    for i, c in enumerate(color_ori):
        new_true[mask == c] = color[i]
    return new_true


def resize_to_short_edge(img: np.ndarray, short_edge_length: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = short_edge_length, int(w * short_edge_length / h)
    else:
        new_w, new_h = short_edge_length, int(h * short_edge_length / w)

    order = 1 if img.ndim == 3 else 0
    factor = 255.0 if img.ndim == 3 else 1.0
    return (factor * resize(img, (new_h, new_w), order=order, anti_aliasing=False)).astype(img.dtype)


def read_image_to_numpy(input_data: Union[Path, str, np.ndarray]) -> np.ndarray:
    """Read image file to numpy array and process it.

    Parameters
    ----------
    input_data : Union[Path, str, np.ndarray]
        The input image data, which can be a file path (string or Path object) or a numpy array.

    Returns
    -------
    np.ndarray
        The processed image as a numpy array with shape (H, W, 3) and dtype uint8.
    """
    # Check if the input data is already a numpy array
    if isinstance(input_data, np.ndarray):
        image = input_data
    elif isinstance(input_data, (str, Path)):
        image = read_file_to_numpy(input_data)
    else:
        raise ValueError(f"Unknown input data type: {type(input_data)}")
    image = image.squeeze()
    # If the image is grayscale (2D array)
    if image.ndim == 2:
        image = normalize_to_uint8(image)
        # Convert grayscale to RGB by stacking the grayscale image into three channels
        image = np.stack([image] * 3, axis=-1)

    # If the image is RGB (3D array with the last dimension size 3)
    elif image.ndim == 3 and image.shape[-1] == 3:
        # Get the indices of channels that are not entirely zero
        non_empty_channels = get_non_empty_channels(image)
        if len(non_empty_channels) == 0:
            # Raise an error if all channels are empty (image is completely black)
            raise ValueError(f"Image shape is {image.shape}, but all pixels are zero.")
        elif len(non_empty_channels) == 1:
            # If only one channel has data, use it for all RGB channels
            c = non_empty_channels[0]
            image = np.stack([normalize_to_uint8(image[:, :, c])] * 3, axis=-1)
        elif len(non_empty_channels) == 2:
            # If two channels have data
            c0, c1 = non_empty_channels
            if np.all(image[:, :, c0] == image[:, :, c1]):
                # If the two channels are identical, use one channel for all RGB channels
                image = np.stack([normalize_to_uint8(image[:, :, c0])] * 3, axis=-1)
            else:
                # Normalize each non-empty channel individually
                image[:, :, c0] = normalize_to_uint8(image[:, :, c0])
                image[:, :, c1] = normalize_to_uint8(image[:, :, c1])
        elif len(non_empty_channels) == 3:
            # If all three channels have data, normalize the entire image
            image = normalize_to_uint8(image)
    else:
        raise ValueError(f"Unknown image shape: {image.shape}")
    return image


def read_mask_to_numpy(input_data: Union[Path, str, np.ndarray]) -> np.ndarray:
    if isinstance(input_data, np.ndarray):
        mask = input_data
    elif isinstance(input_data, (str, Path)):
        mask = read_file_to_numpy(input_data)
    mask = mask.squeeze()
    return remap_mask_color(mask)


def read_file_to_numpy(file: Union[Path, str]) -> np.ndarray:
    file = Path(file)
    suffix = file.suffix.lower()
    full_suffix = "".join(file.suffixes).lower()

    if suffix in [".tif", ".tiff"]:
        return tifffile.imread(file)
    elif suffix in [".bmp", ".png", ".jpeg", ".jpg"]:
        return np.array(Image.open(file))
    elif suffix == ".npy":
        return np.load(file)
    elif suffix == ".nii" or full_suffix == ".nii.gz":
        return nib.load(file).get_fdata()
    else:
        raise NotImplementedError(f"Unknown file type: {file}")


def load_data(
    image_dir: Path, mask_dir: Path, train_id: List[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    image_files = sorted(Path(image_dir).iterdir())
    mask_files = sorted(Path(mask_dir).iterdir())

    if train_id is not None:
        image_files = [image_files[i] for i in train_id]
        mask_files = [mask_files[i] for i in train_id]

    assert len(image_files) == len(mask_files)

    images = [read_image_to_numpy(file) for file in image_files]
    masks = [read_mask_to_numpy(file) for file in mask_files]

    image_file_names = [file.name for file in image_files]
    mask_file_names = [file.name for file in mask_files]

    return images, masks, image_file_names, mask_file_names
