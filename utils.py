import torch
import torch.nn as nn

class temp_eval:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_mode = None

    def __enter__(self):
        self.original_mode = self.model.training
        self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_mode:
            self.model.train()
        else:
            self.model.eval()


def get_model_params(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1024 / 1024
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return dict(
        total_params=total_params,
        total_params_m=total_params_m,
        trainable_params=trainable_params
    )