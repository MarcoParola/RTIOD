import os
import traceback

import torch

from torch.utils.data import WeightedRandomSampler
import numpy as np

def cast2Float(x):
    if isinstance(x, list):
        return [cast2Float(y) for y in x]
    elif isinstance(x, dict):
        return {k: cast2Float(v) for k, v in x.items()}
    return x.float()

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return epoch, best_map, and full checkpoint data"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        best_map = checkpoint.get('best_map', 0.0)

        return epoch + 1, best_map, checkpoint  # Return full checkpoint data

    except Exception as e:
        return 0, 0.0, {}

def adapt_state_dict(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():
        # Handle the specific mismatch
        if key == "class_embed.layers.11.weight":
            new_state_dict["class_embed.classifier.weight"] = value
        elif key == "class_embed.layers.11.bias":
            new_state_dict["class_embed.classifier.bias"] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

def load_weights(args, model, architecture, weight_path, device):
    def filter_head(state_dict):
        # Exclude detection head weights if fine-tuning YOLO
        return {k: v for k, v in state_dict.items() if not k.startswith('head') and not '.head.' in k}

    model_path = None
    exclude_head = args.model == "yolo" and getattr(args.yolo, "finetuning", False)

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        adapted_state_dict = adapt_state_dict(state_dict)
        
        if exclude_head:
            print("Fine-tuning YOLO, excluding detection head from loading weights")
            adapted_state_dict = filter_head(adapted_state_dict)
        
        model.load_state_dict(adapted_state_dict, strict=False)           

    # NOTE J 20/08, args.backbone does not exists and it is accessed in download_full_model_from_huggingface
    # else:
    #     try:
    #         model, model_path = download_full_model_from_huggingface(args, architecture)
    #         download_successful = True
    #         state_dict = torch.load(model_path, map_location=device)
    #         adapted_state_dict = adapt_state_dict(state_dict)
    #         if exclude_head:
    #             adapted_state_dict = filter_head(adapted_state_dict)

    #         model.load_state_dict(adapted_state_dict, strict=False)
    #     except Exception:
    #         traceback.print_exc()

    return model


