"""
Here we define the EfficientViTSAM model class based on the official EfficientViT repository,
but with utils from original CellSeg1 model to make them compatible for our training procedure.
"""

from typing import Any, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.models.efficientvit.sam import EfficientViTSamImageEncoder, PromptEncoder, \
     MaskDecoder, SamResize, SamPad


class EfficientViTSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EfficientViTSamImageEncoder,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        image_size: tuple[int, int] = (512, 512),
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = image_size
        self.image_encoder.img_size = image_size[0]
        self.device = torch.device(device)
        # self.register_buffer("pixel_mean",
        #                      torch.Tensor([123.675 / 255, 116.28 / 255, 103.53 / 255]).view(-1, 1, 1),
        #                      False)
        # self.register_buffer("pixel_std",
        #                      torch.Tensor([58.395 / 255, 57.12 / 255, 57.375 / 255]).view(-1, 1, 1),
        #                      False)

        self.transform = transforms.Compose(
            [
                SamResize(self.image_size[1]),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                    std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
                ),
                SamPad(self.image_size[1]),
            ]
        )

    # @property
    # def device(self) -> Any:
    #     return self.pixel_mean.device

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: tuple[int, ...],
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_size[0], self.image_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    # def forward(
    #     self,
    #     batched_input: list[dict[str, Any]],
    #     multimask_output: bool,
    # ):
    #     input_images = torch.stack([x["image"] for x in batched_input], dim=0)

    #     image_embeddings = self.image_encoder(input_images)

    #     outputs = []
    #     iou_outputs = []
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
    #         outputs.append(low_res_masks)
    #         iou_outputs.append(iou_predictions)

    #     outputs = torch.stack([out for out in outputs], dim=0)
    #     iou_outputs = torch.stack(iou_outputs, dim=0)

    #     return outputs, iou_outputs

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # # Normalize colors
        # x = (x - self.pixel_mean) / self.pixel_std

        # # Pad
        # h, w = x.shape[-2:]
        # padh = self.image_encoder.img_size - h
        # padw = self.image_encoder.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return self.transform(x)
