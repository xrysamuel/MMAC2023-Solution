
from typing import Optional
import threading

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

def get_module_param_names(module: nn.Module):
    all_param_names = []
    trainable_param_names = []

    for name, param in module.named_parameters():
        all_param_names.append(name)
        if param.requires_grad:
            trainable_param_names.append(name)

    return {
        "All parameter names": all_param_names,
        "Trainable parameter names": trainable_param_names
    }


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

class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        self.original_layer = original_layer
        # Freeze the original layer's parameters
        self.original_layer.weight.requires_grad_(False)
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad_(False)

        self.rank = rank
        self.lora_alpha = lora_alpha
        # Pre-calculate scaling factor
        self.scaling = self.lora_alpha / self.rank

        # LoRA A matrix (in_features, rank)
        # Bias is not needed for LoRA matrices
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        # LoRA B matrix (rank, out_features)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)

        # LoRA Dropout: Applied to the input 'x' before it enters the LoRA path (A -> B).
        # This helps in regularization during fine-tuning.
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Initialize LoRA A with Kaiming uniform and LoRA B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer's forward pass
        original_output = self.original_layer(x)

        # Apply LoRA dropout to the input 'x' before the LoRA path.
        x_lora = self.lora_dropout(x)

        # LoRA's forward pass: x @ lora_A @ lora_B * scaling
        lora_output = self.lora_B(self.lora_A(x_lora)) * self.scaling

        return original_output + lora_output



def apply_lora_to_model(model: nn.Module, rank: int = 64, lora_alpha: float = 128, 
                        lora_dropout: float = 0.0, exclude_modules: Optional[list] = None):
    """
    Applies LoRA (Low-Rank Adaptation) to all eligible nn.Linear layers within a PyTorch model.
    It freezes all parameters of the original model first. Then, for each eligible `nn.Linear` layer,
    it replaces it with a `LoRALayer` instance. The `LoRALayer` internally manages the frozen
    original linear layer and adds small, trainable low-rank adaptation matrices (LoRA A and LoRA B),
    along with an optional LoRA dropout. This means that after this function is called,
    only the LoRA matrices will be trainable.

    Args:
        model (nn.Module): The PyTorch model to apply LoRA to.
        rank (int): The rank 'r' of the LoRA update matrices. A higher rank allows for more expressiveness
                    but increases trainable parameters. Defaults to 64.
        lora_alpha (float): The LoRA scaling factor 'alpha'. This value is used to scale the output of the
                            LoRA path: `scaling = lora_alpha / rank`. Defaults to 128.
        lora_dropout (float): The dropout probability applied to the input of the LoRA path in `LoRALayer`.
                              Defaults to 0.0 (no dropout).
        exclude_modules (Optional[list]): A list of strings. If a module's full hierarchical name
                                          (e.g., 'encoder.layer.0.attn.q_proj') contains any of
                                          these strings as a substring, that `nn.Linear` layer
                                          will be excluded from LoRA adaptation. Its parameters
                                          will remain frozen along with the rest of the original model.
                                          Defaults to None (no exclusions).
    """
    if exclude_modules is None:
        exclude_modules = []

    # Freeze all parameters in the entire model first.
    for param in model.parameters():
        param.requires_grad_(False)

    # Iterate through named modules to find nn.Linear layers and replace them with LoRALayer.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_exclude = False
            # Check if the current module's name should be excluded from LoRA adaptation.
            for exclude_name_part in exclude_modules:
                if exclude_name_part in name:
                    should_exclude = True
                    break

            if should_exclude:
                # If excluded, skip this module and proceed to the next one.
                continue

            # Get the parent module to replace the child.
            # This handles both nested modules (e.g., model.encoder.layer.0.attn.q_proj)
            # and top-level modules (e.g., model.linear_layer).
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
                child_name = name

            # Replace the original nn.Linear layer with our LoRALayer.
            # The LoRALayer will internally manage the frozen original layer and its own trainable LoRA matrices.
            setattr(parent_module, child_name, LoRALayer(module, rank, lora_alpha, lora_dropout))


class interrupt_proof:
    def __init__(self, graceful_exit_function):
        if not callable(graceful_exit_function):
            raise ValueError("graceful_exit_function must be a callable.")
        self.graceful_exit_function = graceful_exit_function

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            self.graceful_exit_function()
            return True  # Suppress KeyboardInterrupt
        return False  # Propagate other exceptions