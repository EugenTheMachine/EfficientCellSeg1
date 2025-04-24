# Modified from Segment Anything Model (SAM)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    # def forward(
    #     self,
    #     batched_input: List[Dict[str, Any]],
    #     multimask_output: bool,
    # ) -> List[Dict[str, torch.Tensor]]:
    #     input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
    #     image_embeddings = self.image_encoder(input_images)

    #     outputs = []
    #     for image_record, curr_embedding in zip(batched_input, image_embeddings):
    #         if "point_coords" in image_record:
    #             points = (image_record["point_coords"], image_record["point_labels"])
    #         else:
    #             points = None
    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=points,
    #             boxes=image_record.get("boxes", None),
    #             masks=image_record.get("mask_inputs", None),
    #         )
    #         low_res_masks, iou_predictions = self.mask_decoder(
    #             image_embeddings=curr_embedding.unsqueeze(0),
    #             image_pe=self.prompt_encoder.get_dense_pe(),
    #             sparse_prompt_embeddings=sparse_embeddings,
    #             dense_prompt_embeddings=dense_embeddings,
    #             multimask_output=multimask_output,
    #         )
    #         masks = self.postprocess_masks(
    #             low_res_masks,
    #             input_size=image_record["image"].shape[-2:],
    #             original_size=image_record["original_size"],
    #         )
    #         masks = masks > self.mask_threshold
    #         outputs.append(
    #             {
    #                 "masks": masks,
    #                 "iou_predictions": iou_predictions,
    #                 "low_res_logits": low_res_masks,
    #             }
    #         )
    #     return outputs

    def encoder_image_embeddings(self, images: List[torch.Tensor],):
        input_images = torch.stack([self.preprocess(x) for x in images], dim=0)
        image_embeddings = self.image_encoder(input_images)
        return image_embeddings

    def forward_train(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        image_size: Tuple[int, ...],
        input_image_embeddings: torch.Tensor = None,
    ) -> List[Dict[str, torch.Tensor]]:

        image_embeddings = input_image_embeddings
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            points = (image_record["point_coords"], image_record["point_labels"])
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_size,
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
