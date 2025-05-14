# Modified from Segment Anything Model (SAM)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0
from typing import Callable, Optional
from functools import partial

import torch
from torch.nn import functional as F

from seg_any.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
from seg_any.modeling.evit import EfficientViTSam, EfficientViTSamImageEncoder

from efficientvit.models.efficientvit.sam import SamNeck, build_kwargs_from_config
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file


def build_efficientvit_sam(image_encoder: EfficientViTSamImageEncoder, image_size: int) -> EfficientViTSam:
    return EfficientViTSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(512, 512),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        image_size=(512, image_size),
    )


def efficientvit_sam_l0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l0

    backbone = efficientvit_backbone_l0(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=4,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l1(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_l2(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl0(image_size: int = 512, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[0, 1, 1, 2, 3, 3],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=6,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)


def efficientvit_sam_xl1(image_size: int = 1024, **kwargs) -> EfficientViTSam:
    from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone

    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 4, 6, 6],
        block_list=["res", "fmb", "fmb", "fmb", "att@3", "att@3"],
        expand_list=[1, 4, 4, 4, 4, 6],
        fewer_norm_list=[False, False, False, False, True, True],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )

    neck = SamNeck(
        fid_list=["stage5", "stage4", "stage3"],
        in_channel_list=[1024, 512, 256],
        head_width=256,
        head_depth=12,
        expand_ratio=4,
        middle_op="fmb",
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    return build_efficientvit_sam(image_encoder, image_size)

REGISTERED_EFFICIENTVIT_SAM_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-sam-l0": (efficientvit_sam_l0, 1e-6, "efficientvit_sam/efficientvit_sam_l0.pt"),
    "efficientvit-sam-l1": (efficientvit_sam_l1, 1e-6, "efficientvit_sam/efficientvit_sam_l1.pt"),
    "efficientvit-sam-l2": (efficientvit_sam_l2, 1e-6, "efficientvit_sam/efficientvit_sam_l2.pt"),
    "efficientvit-sam-xl0": (efficientvit_sam_xl0, 1e-6, "efficientvit_sam/efficientvit_sam_xl0.pt"),
    "efficientvit-sam-xl1": (efficientvit_sam_xl1, 1e-6, "efficientvit_sam/efficientvit_sam_xl1.pt"),
}

def create_efficientvit_sam_model(
    name: str, pretrained=True, weight_url: Optional[str] = None, **kwargs
) -> EfficientViTSam:
    if name not in REGISTERED_EFFICIENTVIT_SAM_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SAM_MODEL.keys())}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[name]
        model = model_cls(**kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model


def build_sam_vit_h(checkpoint=None, image_size=512):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=image_size,
        checkpoint=checkpoint,
    )


def build_sam_vit_l(checkpoint=None, image_size=512):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=image_size,
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None, image_size=512):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=image_size,
        checkpoint=checkpoint,
    )

def build_esam(checkpoint=None, image_size=512):
    return create_efficientvit_sam_model(
        name="efficientvit-sam-l0",
        weight_url=checkpoint
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "evit": build_esam,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint=None,
):
    prompt_embed_dim = 256
    # image_size = 512
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, weights_only=True)
        try:
            sam.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = load_from(
                sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes
            )
            sam.load_state_dict(new_state_dict)
    return sam


def check_contain(key, except_strings):
    for strs in except_strings:
        if strs in key:
            return True
    return False


# https://github.com/hitachinsk/SAMed/blob/main/segment_anything/build_sam.py

# MIT License

# Copyright (c) 2023 Kaidong Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    sam_dict = sam.state_dict()
    except_strings = []
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in sam_dict.keys() and not check_contain(k, except_strings):
            new_state_dict[k] = v
    pos_embed = new_state_dict["image_encoder.pos_embed"]
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(
            pos_embed, (token_size, token_size), mode="bilinear", align_corners=True
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict["image_encoder.pos_embed"] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if "rel_pos" in k]
        global_rel_pos_keys = []
        for k in rel_pos_keys:
            flag = any(
                [(str(j) == k.split(".")[2]) for j in encoder_global_attn_indexes]
            )
            if flag:
                global_rel_pos_keys.append(k)
        # global_rel_pos_keys = [k for k in rel_pos_keys if "2" in k or "5" in k or "8" in k or "11" in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(
                rel_pos_params,
                (token_size * 2 - 1, w),
                mode="bilinear",
                align_corners=True,
            )
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict


if __name__ == "__main__":
    pass
