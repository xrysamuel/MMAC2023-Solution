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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Total Parameters (M)": round(total_params / 1_000_000, 2), # Converted to Millions, rounded to 2 decimal places
        "Trainable Parameters (M)": round(trainable_params / 1_000_000, 2),
        "Total Parameters (raw)": total_params,
        "Trainable Parameters (raw)": trainable_params,
        "Percentage Trainable": f"{round((trainable_params / total_params) * 100, 2)}%" if total_params > 0 else "0%"
    }