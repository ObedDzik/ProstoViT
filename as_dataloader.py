import json
import torch
from torch.utils.data import DataLoader
import os


def make_loader(*, dataset, cfg, shuffle, **kwargs):
    batch_size=kwargs.get("batch_size", getattr(cfg, "batch_size", 16))
    num_workers=kwargs.get("num_workers", getattr(cfg, "num_workers", 4))
    pin_memory=kwargs.get("pin_memory", getattr(cfg, "pin_memory", True))
    weighted=kwargs.get("weighted", getattr(cfg, "weighted", False))
    batch_sampler=kwargs.get('batch_sampler', False)

    if weighted:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



