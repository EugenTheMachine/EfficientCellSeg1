from seg_any import sam_model_registry


def count_parameters(model_type):
    sam = sam_model_registry[model_type](checkpoint=None)
    total_params = sum(p.numel() for p in sam.parameters())
    print(f"{model_type} total parameters: {total_params:,}")
    return total_params


for model_type in ["vit_h", "vit_l", "vit_b"]:
    count_parameters(model_type)
