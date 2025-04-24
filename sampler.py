import math
import random
from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
from scipy.ndimage import find_objects

from data.utils import remap_mask_color


def sample_2d(distance, num_samples, equal_prob=False):
    distance_flat = distance.flatten()
    indices = np.arange(len(distance_flat))
    probabilities = distance_flat / np.sum(distance_flat)
    if equal_prob:
        probabilities[probabilities > 0] = 1.0
        probabilities = probabilities / np.sum(probabilities)
    chosen_indices = np.random.choice(indices, p=probabilities, size=num_samples)
    sampled_points = [np.unravel_index(idx, distance.shape) for idx in chosen_indices]
    return sampled_points


def filter_small_distance(distance, area_ratio):
    assert 0.0 < area_ratio <= 1.0
    dist = deepcopy(distance)
    area = np.sum(dist != 0)
    threshold = np.sort(dist.flatten())[-int(area * area_ratio)]
    dist[dist < threshold] = 0
    return dist


def zero_edge(mask, edge_dist):
    mask = deepcopy(mask)
    mask[:edge_dist, :] = 0
    mask[:, :edge_dist] = 0
    mask[-edge_dist:, :] = 0
    mask[:, -edge_dist:] = 0
    return mask


def sample_points(
    ins_mask,
    pos_rate=1.0,
    neg_rate=0.5,
    neg_area_ratio_threshold=5,
    neg_area_threshold=1000,
    max_point_num=100,
    edge=10,
    min_cell_area=20,
    foreground_sample_area_ratio=0.5,
    background_sample_area_ratio=0.8,
    foreground_equal_prob=True,
    background_equal_prob=False,
):
    assert 0.0 <= pos_rate <= 1.0
    assert 0.0 <= neg_rate <= 1.0
    assert 0.0 < foreground_sample_area_ratio <= 1.0
    assert 0.0 < background_sample_area_ratio <= 1.0

    ins_mask = remap_mask_color(ins_mask, random=False)
    ins_mask_pad = np.pad(ins_mask, edge, mode="constant", constant_values=0)
    ins_mask_pad_1 = np.pad(ins_mask, edge, mode="constant", constant_values=1)

    mask_max = ins_mask_pad.max()
    pos_num = min(max_point_num, math.ceil(mask_max * pos_rate))
    neg_num = min(max_point_num, math.ceil(mask_max * neg_rate))

    pos_area = np.sum(ins_mask != 0)
    neg_area = np.sum(ins_mask == 0)
    area_ratio = pos_area / neg_area
    if area_ratio > neg_area_ratio_threshold:
        neg_num = 0

    if (pos_num == 0) and (neg_num == 0):
        neg_num = 1

    background_dist, labels = cv2.distanceTransformWithLabels(
        (ins_mask_pad_1 == 0).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    slices = find_objects(ins_mask_pad)
    idx = random.sample(range(len(slices)), pos_num)
    slices = [slices[i] for i in idx]

    pos_points = []
    neg_points = []
    pos_sample_map = np.zeros_like(ins_mask_pad, dtype=np.float32)

    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            cell_mask = ins_mask_pad[sr, sc] == (idx[i] + 1)

            if check_at_edge(si, 2 * edge, ins_mask_pad.shape):
                continue

            cell_area = np.sum(cell_mask)
            if cell_area <= min_cell_area:
                continue

            pos_dist_map, labels = cv2.distanceTransformWithLabels(
                cell_mask.astype(np.uint8),
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            pos_dist_map[~cell_mask] = 0
            pos_dist_map = filter_small_distance(pos_dist_map, foreground_sample_area_ratio)
            point_in_slice = sample_2d(pos_dist_map, 1, foreground_equal_prob)[0]
            pos_sample_map[sr, sc] += pos_dist_map

            point = [point_in_slice[1] + sc.start, point_in_slice[0] + sr.start]
            pos_points.append(point)

    neg_sample_map = filter_small_distance(background_dist, background_sample_area_ratio)
    neg_area = np.sum(neg_sample_map != 0)

    neg_num = min(neg_num, int(neg_area / neg_area_threshold) + 1)

    if neg_area != 0:
        neg_points = sample_2d(neg_sample_map, neg_num, background_equal_prob)
    else:
        neg_points = []
        neg_sample_map = np.zeros_like(ins_mask_pad)
    neg_points = [[p[1], p[0]] for p in neg_points]

    points = np.array(pos_points + neg_points)
    types = np.zeros(len(points), dtype=np.uint16)
    types[0 : len(pos_points)] = 1

    points -= edge
    pos_sample_map = pos_sample_map[edge:-edge, edge:-edge]
    neg_sample_map = neg_sample_map[edge:-edge, edge:-edge]
    if foreground_equal_prob:
        pos_sample_map[pos_sample_map > 0] = 1.0
    if background_equal_prob:
        neg_sample_map[neg_sample_map > 0] = 1.0

    return points, types, pos_sample_map, neg_sample_map


def check_at_edge(si, edge_distance, image_shape):
    sr, sc = si
    if sr.stop < edge_distance:
        return True
    if sc.stop < edge_distance:
        return True
    if sr.start > image_shape[0] - edge_distance:
        return True
    if sc.start > image_shape[1] - edge_distance:
        return True
    return False


def create_collate_fn(config):
    def custom_collate_fn(batch):
        images = []
        instance_masks = []
        cell_masks = []
        all_points = []
        all_types = []
        size = config["sam_image_size"]
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=config["bright_limit"],
                    contrast_limit=config["contrast_limit"],
                    p=config["bright_prob"],
                ),
                # A.Flip(p=config["flip_prob"]),
                A.HorizontalFlip(p=config["flip_prob"]),
                A.VerticalFlip(p=config["flip_prob"]),
                A.RandomResizedCrop(
                    height=size,
                    width=size,
                    scale=config["crop_scale"],
                    ratio=config["crop_ratio"],
                    p=config["crop_prob"],
                    interpolation=cv2.INTER_LINEAR_EXACT,
                ),
                A.ShiftScaleRotate(
                    scale_limit=config["scale_limit"],
                    p=config["rotate_prob"],
                    border_mode=cv2.BORDER_CONSTANT,
                    interpolation=cv2.INTER_LINEAR,
                ),
                A.Resize(height=size, width=size),
            ],
            keypoint_params=A.KeypointParams(format="xy", label_fields=["idx"]),
        )

        for image, instance_mask in batch:
            org_points, org_types, pos_map, neg_map = sample_points(
                instance_mask,
                pos_rate=config["pos_rate"],
                neg_rate=config["neg_rate"],
                neg_area_ratio_threshold=config["neg_area_ratio_threshold"],
                neg_area_threshold=config["neg_area_threshold"],
                max_point_num=config["max_point_num"],
                edge=config["edge_distance"],
                min_cell_area=config["min_cell_area"],
                foreground_sample_area_ratio=config["foreground_sample_area_ratio"],
                background_sample_area_ratio=config["background_sample_area_ratio"],
                foreground_equal_prob=config["foreground_equal_prob"],
                background_equal_prob=config["background_equal_prob"],
            )

            if config["data_augmentation"]:
                restore_to_no_augment = False
                idx = list(range(len(org_points)))

                transformed = transform(image=image, mask=instance_mask, keypoints=org_points, idx=idx)
                t_image = transformed["image"]
                t_instance_mask = transformed["mask"]
                t_points = np.array(transformed["keypoints"]).astype(np.uint16)
                t_idx = np.array(transformed["idx"]).astype(np.int32)
                if len(t_idx) == 0:
                    restore_to_no_augment = True
                    continue
                t_types = org_types[t_idx]
                t_pos_points = t_points[t_types == 1]
                t_neg_points = t_points[t_types == 0]

                pos_cell_mask = []
                delete_idx = []
                for i, point in enumerate(t_pos_points):
                    mask = np.zeros_like(t_instance_mask, dtype=np.uint16)
                    point_label = t_instance_mask[point[1], point[0]]
                    if point_label == 0:
                        delete_idx.append(i)
                        continue
                    mask[t_instance_mask == point_label] = 1
                    cell_area = np.sum(mask)
                    if cell_area <= config["min_cell_area"]:
                        delete_idx.append(i)
                        continue
                    slices = find_objects(mask)
                    assert slices is not None
                    assert len(slices) == 1
                    si = slices[0]
                    if not check_at_edge(si, config["edge_distance"], mask.shape):
                        pos_cell_mask.append(mask)
                    else:
                        delete_idx.append(i)
                t_points = np.delete(t_points, delete_idx, axis=0)
                t_types = np.delete(t_types, delete_idx, axis=0)

            if (not config["data_augmentation"]) or restore_to_no_augment:
                t_image = image
                t_instance_mask = instance_mask
                t_points = org_points
                t_types = org_types
                t_pos_points = t_points[t_types == 1]
                t_neg_points = t_points[t_types == 0]

                pos_cell_mask = []
                for point in t_pos_points:
                    mask = np.zeros_like(t_instance_mask, dtype=np.uint16)
                    point_label = t_instance_mask[point[1], point[0]]
                    assert point_label != 0
                    mask[t_instance_mask == point_label] = 1
                    pos_cell_mask.append(mask)

            neg_cell_mask = [np.zeros_like(t_instance_mask, dtype=np.uint16) for i in range(len(t_neg_points))]

            images.append(t_image.astype(np.uint8))
            cell_masks.append(pos_cell_mask + neg_cell_mask)
            instance_masks.append(t_instance_mask.astype(np.uint16))
            all_points.append(t_points)
            all_types.append(t_types)

        return (
            images,
            instance_masks,
            cell_masks,
            all_points,
            all_types,
        )

    return custom_collate_fn
