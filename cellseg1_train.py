import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
# from tqdm import tqdm
from tqdm.notebook import tqdm

from cell_loss import cell_prob_mse_loss, cross_entropy_loss
from data.dataset import TrainDataset
from gpu_memory_tracker import GPUMemoryTracker
from peft.sam_lora_image_encoder_mask_decoder import LoRa_ESAM
from sampler import create_collate_fn
from seg_any import sam_model_registry
# from mobile_sam import sam_model_registry
from set_environment import set_env
from predict import load_model_from_config
import matplotlib.pyplot as plt


def prepare_directories(config: Dict):
    Path(config["result_pth_path"]).parent.mkdir(exist_ok=True, parents=True)


def load_dataset(config: Dict) -> TrainDataset:
    return TrainDataset(
        image_dir=Path(config["train_image_dir"]),
        mask_dir=Path(config["train_mask_dir"]),
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )


def load_eval_dataset(config: Dict) -> TrainDataset:
    return TrainDataset(
        image_dir=Path(config["test_image_dir"]),
        mask_dir=Path(config["test_mask_dir"]),
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )


def load_model(config: Dict) -> LoRa_ESAM:
    model = sam_model_registry[config["vit_name"]](checkpoint=config["model_path"],
                                                   image_size=config["sam_image_size"])
    return LoRa_ESAM(model, config).cuda()


def setup_training(
    config: Dict, model: LoRa_ESAM, train_dataset: TrainDataset, test_dataset: TrainDataset = None
) -> Tuple[DataLoader, optim.Optimizer, OneCycleLR]:
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["base_lr"],
    )
    custom_collate_func = create_collate_fn(config)
    trainloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=custom_collate_func,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["base_lr"],
        total_steps=config["epoch_max"]
        * (len(trainloader) + config["gradient_accumulation_step"] - 1)
        // config["gradient_accumulation_step"],
        pct_start=config["onecycle_lr_pct_start"],
    )
    if test_dataset is not None:
        testloader = DataLoader(
            test_dataset,
            batch_size=config["val_batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=custom_collate_func,
        )
        return trainloader, testloader, optimizer, scheduler
    return trainloader, None, optimizer, scheduler


def to_tensor(
    images: List[np.ndarray], all_points: List[List[np.ndarray]], image_size: int
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    tensor_images = [torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float).cuda() for image in images]
    items = [
        {
            "point_coords": torch.as_tensor(np.stack(points).astype(np.int64), dtype=torch.float)[:, None, :].cuda(),
            "point_labels": torch.ones(len(points), 1, dtype=torch.int).cuda(),
            "original_size": (image_size, image_size),
        }
        for points in all_points
    ]
    return tensor_images, items


def extract_outputs(outputs: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_logits = []
    pred_cell_probs = []
    for output in outputs:
        point_nums = output["masks"].shape[0]
        for i in range(point_nums):
            pred_logits.append(output["low_res_logits"][i][0])
            pred_cell_probs.append(output["iou_predictions"][i][0])
    return torch.stack(pred_logits).cuda(), torch.stack(pred_cell_probs).cuda()


def extract_true_masks(
    images: List[np.ndarray],
    cell_masks: List[np.ndarray],
    all_points: List[List[np.ndarray]],
    all_cell_probs: List[List[int]],
    low_res_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    true_masks = []
    true_cell_probs = []
    for image, masks, points, cell_probs in zip(images, cell_masks, all_points, all_cell_probs):
        for mask, point, cell_prob in zip(masks, points, cell_probs):
            low_res_true_mask = cv2.resize(
                mask.astype(np.int32),
                dsize=(low_res_shape[0], low_res_shape[1]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            if low_res_true_mask.max() == 0:
                cell_prob = 0
            true_masks.append(low_res_true_mask)
            true_cell_probs.append(cell_prob)
    true_cell_probs = torch.tensor(true_cell_probs, dtype=torch.float32).cuda()
    true_masks = torch.tensor(np.array(true_masks), dtype=torch.float32).cuda()
    # plt.imshow(true_masks[0].cpu().detach().numpy())
    # plt.title("GT")
    # plt.show()
    return true_masks, true_cell_probs


def is_valid_batch(images: List[np.ndarray], all_points: List[List[np.ndarray]]) -> bool:
    return len(images) > 0 and len(all_points) > 0 and all(len(points) > 0 for points in all_points)


def compute_loss(
    model: LoRa_ESAM,
    config: Dict,
    batch_images: List[torch.Tensor],
    batch_points: List[Dict[str, torch.Tensor]],
    cell_masks: List[np.ndarray],
    all_points: List[List[np.ndarray]],
    all_cell_probs: List[List[int]],
) -> torch.Tensor:
    image_embeddings = model.sam.encoder_image_embeddings(batch_images)
    outputs = model.sam.forward_train(
        batched_input=batch_points,
        multimask_output=False,
        input_image_embeddings=image_embeddings,
        image_size=(config["sam_image_size"], config["sam_image_size"]),
    )

    pred_logits, pred_cell_probs = extract_outputs(outputs)
    # plt.imshow(pred_logits[0].cpu().detach().numpy())
    # plt.show()
    true_masks, true_cell_prob = extract_true_masks(
        batch_images, cell_masks, all_points, all_cell_probs, pred_logits[0].shape
    )

    ce_loss = cross_entropy_loss(
        true_masks=true_masks,
        pred_logits=pred_logits,
        true_cell_prob=true_cell_prob,
    )
    cell_prob_loss = cell_prob_mse_loss(true_cell_prob=true_cell_prob, pred_cell_prob=pred_cell_probs)
    return cell_prob_loss + ce_loss * config["ce_loss_weight"]


def train_epoch(
    model: LoRa_ESAM,
    config: Dict,
    trainloader: DataLoader,
    testloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: OneCycleLR,
    epoch: int,
    val_period: int = 1,
    len_train: int = 1,
    len_test: int = 1,
    stop_event=None,
    scaler=None
):
    model.train()
    actual_ga_step = 0
    # training phase
    total_train_loss = []
    for i_batch, batch_data in enumerate(tqdm(trainloader, desc="Batches", leave=False)):
        if stop_event is not None and stop_event.is_set():
            return
        images, true_instance_masks, cell_masks, all_points, all_cell_probs = batch_data

        if not is_valid_batch(images, all_points):
            continue

        batch_images, batch_points = to_tensor(images, all_points, config["sam_image_size"])
        with autocast(device_type="cuda"):  # Enables mixed precision
            loss = compute_loss(model, config, batch_images, batch_points, cell_masks, all_points, all_cell_probs)
        total_train_loss.append(loss.item())

        actual_ga_step += 1
        loss_ga = loss / (actual_ga_step if (i_batch + 1) == len(trainloader) else config["gradient_accumulation_step"])
        # loss_ga.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ((i_batch + 1) % config["gradient_accumulation_step"] == 0) or ((i_batch + 1) == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
            actual_ga_step = 0
            scheduler.step()
    train_loss = sum(total_train_loss) / len_train
    torch.cuda.empty_cache()
    if epoch % val_period == 0:
        # validation phase (here testloader is in fact validation loader)
        model.eval()
        total_val_loss = []
        # Use torch.no_grad() to disable gradient tracking during validation
        with torch.no_grad():
            for i_batch, batch_data in enumerate(tqdm(testloader, desc="Batches", leave=False)):
                images, true_instance_masks, cell_masks, all_points, all_cell_probs = batch_data
                if not is_valid_batch(images, all_points):
                    continue
                batch_images, batch_points = to_tensor(images, all_points, config["sam_image_size"])
                loss = compute_loss(model, config, batch_images, batch_points, cell_masks, all_points, all_cell_probs)
                total_val_loss.append(loss.item())
        val_loss = sum(total_val_loss) / len_test
    else:
        val_loss = -1
    torch.cuda.empty_cache()
    return train_loss, val_loss


def save_model_pth(model: LoRa_ESAM, save_path: str):
    model.save_lora_parameters(save_path)


def main(config_path: Union[str, Dict, Path], save_model: bool = True) -> LoRa_ESAM:
    if isinstance(config_path, dict):
        config = config_path
    elif isinstance(config_path, str) or isinstance(config_path, Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    prepare_directories(config)

    train_dataset = load_dataset(config)
    test_dataset = load_eval_dataset(config)
    # model = load_model(config)
    try:
        model = load_model_from_config(config, empty_lora=False)
        print("Successfully loaded LoRa checkpoint. Proceeding with it...")
    except:
        print("Failed to load LoRa ckp. Loading raw model instead...")
        model = load_model(config)
        print("Successfully loaded raw model. Proceeding with it...")
    trainloader, testloader, optimizer, scheduler = setup_training(config,
                                                                   model,
                                                                   train_dataset,
                                                                   test_dataset)
    if config["track_gpu_memory"]:
        gpu_memory_tracker = GPUMemoryTracker()
        gpu_memory_tracker.reset()
        memory_stats = {}
    best_val_loss = float("inf")
    patience = config['patience']
    current_patience = 0
    training_log = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }
    # saving cfg file beforehead
    with open(config["result_dir"] + "/config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    scaler = GradScaler(device="cuda")
    # proceeding with actual training
    for epoch in tqdm(range(config["epoch_max"]), desc="Epochs"):
        train_loss, val_loss = train_epoch(model, config, trainloader, testloader, optimizer,
                                           scheduler, epoch=epoch, val_period=config['val_period'],
                                           len_train=config['len_train'], len_test=config['len_test'],
                                           scaler=scaler)
        training_log["train_loss"].append(train_loss)
        training_log["val_loss"].append(val_loss)
        training_log["epoch"].append(epoch)
        if save_model or True:
            save_path = Path(config["result_pth_path"]).parent / f"sam_lora_epoch_{str(epoch).zfill(2)}.pth"
            save_model_pth(model, str(save_path))
            csv_path = Path(config["result_dir"]).parent / "training_log.csv"
            pd.DataFrame(training_log).to_csv(csv_path, index=False)
            print(f"Training log saved to {csv_path}")
        if val_loss != -1 and val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
            if save_model or True:
                save_path = Path(config["result_pth_path"]).parent / f"sam_lora_best_epoch_{str(epoch).zfill(2)}.pth"
                save_model_pth(model, str(save_path))
        else:
            current_patience += 1 / config['val_period']
            if current_patience >= patience:
                print(f"Early stopping at epoch {int(epoch)}. Best val loss: {best_val_loss}")
                break
        if config["track_gpu_memory"]:
            memory_stats[epoch] = gpu_memory_tracker.get_memory_stats()
    csv_path = Path(config["result_dir"]).parent / "training_log.csv"
    pd.DataFrame(training_log).to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")
    if save_model or True:
        save_model_pth(model, config["result_pth_path"])
    
    if config["track_gpu_memory"]:
        with open(Path(config["result_pth_path"]).parent / "memory_stats.json", "w") as f:
            json.dump(memory_stats, f, indent=4)
    return model
