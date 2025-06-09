import os
import logging
from typing import Optional
from functools import partial
import cv2
import torch
from torch import nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class model:
    def __init__(self):
        self.checkpoint = "retfound_best.pth"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")
        
        # Define ImageNet standard mean and standard deviation as tensors
        # These are for channels in RGB order
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.output_size = (224, 224)

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.model = RETFoundMAEClassifier(num_classes=5, pretrained=False)
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_image, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: an int value indicating the class for the input image.
        """
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = CenterCropTransform()(image)
        image = cv2.resize(image, self.output_size)
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = (image - self.imagenet_mean) / self.imagenet_std
        image = image.unsqueeze(0)
        image = image.to(self.device, torch.float)

        with torch.no_grad():
            score = self.model(image)
        _, pred_class = torch.max(score, 1)
        pred_class = pred_class.detach().cpu()

        return int(pred_class)


class CenterCropTransform:
    def __init__(self, size: int = 224, scale_size: int = 256):
        """
        Initializes the center crop transformation.
        Args:
            size (int): The final square side length of the cropped image, default is 224.
            scale_size (int): The size to which the shorter side of the image will be scaled
                              before cropping, default is 256.
        """
        self.size = size
        self.scale_size = scale_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the center crop transformation to an image.
        Args:
            image (np.ndarray): Input image in HWC format, 0-255 range.
        Returns:
            np.ndarray: Transformed image in HWC format, 0-255 range.
        """
        h, w, _ = image.shape

        # 1. Scale the shorter side to self.scale_size, maintaining aspect ratio for the longer side.
        if h < w:
            new_h = self.scale_size
            new_w = int(w * (self.scale_size / h))
        else:
            new_w = self.scale_size
            new_h = int(h * (self.scale_size / w))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Center crop to self.size x self.size.
        h_resized, w_resized, _ = image.shape
        top = (h_resized - self.size) // 2
        left = (w_resized - self.size) // 2
        # Ensure crop region doesn't go out of bounds
        top = max(0, top)
        left = max(0, left)
        image = image[top:top + self.size, left:left + self.size, :]
        return image


class RETFoundMAEClassifier(nn.Module):
    import timm

    class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
        """Vision Transformer with support for global average pooling"""

        def __init__(self, num_classes, **kwargs):
            kwargs.update(
                dict(
                    img_size=224,
                    patch_size=16,
                    embed_dim=1024,
                    depth=24,
                    drop_path_rate=0.2,
                    num_heads=16,
                    num_classes=num_classes,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
            )
            super().__init__(**kwargs)

            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        def forward_features(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = x[:, 1:, :].mean(
                dim=1, keepdim=True
            )  # global pool without cls token
            outcome = self.fc_norm(x)

            return outcome

    def __init__(self, num_classes: int = 5, pretrained: bool = True, lora_backbone: bool = True):
        super().__init__()
        from timm.models.layers import trunc_normal_

        if pretrained:
            self.retfoundmae = self.VisionTransformer(num_classes)
            logger.info("Attempting to download model from Hugging Face Hub.")
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id='YukunZhou/RETFound_mae_natureCFP',
                filename='RETFound_mae_natureCFP.pth',
                local_dir='.cache/'
            )
            logger.info(f"Loading model checkpoint from: {checkpoint_path}.")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_model = checkpoint['model']
            checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
            checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
            state_dict = self.retfoundmae.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    logger.warning(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self._interpolate_pos_embed(self.retfoundmae, checkpoint_model)
            self.retfoundmae.load_state_dict(checkpoint_model, strict=False)
            trunc_normal_(self.retfoundmae.head.weight, std=2e-5)
            logger.info(
                "Initialized RETFoundMAE with ImageNet pre-trained weights."
            )
        else:
            self.retfoundmae = self.VisionTransformer(num_classes)
            state_dict = self.retfoundmae.state_dict()
            trunc_normal_(self.retfoundmae.head.weight, std=2e-5)
            logger.info("Initialized RETFoundMAE without pre-trained weights.")

        if lora_backbone:
            apply_lora_to_model(self.retfoundmae, rank=16, lora_alpha=32, lora_dropout=0.1, exclude_modules=["head"])
            logger.info(f"Apply lora to model.")
            for name, param in self.retfoundmae.named_parameters():
                if "head" in name:
                    param.requires_grad = True
                    logger.info(f"Parameter {name} is trainable.")


    def _interpolate_pos_embed(self, model, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                logger.info("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.retfoundmae(x)
    


class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, lora_alpha: float):
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

        # Initialize LoRA A with Kaiming uniform and LoRA B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer's forward pass
        original_output = self.original_layer(x)

        # LoRA's forward pass: x @ lora_A @ lora_B * scaling
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling

        return original_output + lora_output



def apply_lora_to_model(model: nn.Module, rank: int = 64, lora_alpha: float = 128, exclude_modules: Optional[list] = None):
    """
    Applies LoRA (Low-Rank Adaptation) to all nn.Linear layers within a PyTorch model,
    excluding specified modules.

    This function first **freezes all parameters** of the original model. Then, for each
    eligible `nn.Linear` layer, it replaces it with a `LoRALayer` instance. The `LoRALayer`
    internally manages the frozen original linear layer and adds small, trainable
    low-rank adaptation matrices (LoRA A and LoRA B). This means that after this
    function is called, **only the LoRA matrices will be trainable**.

    Args:
        model (nn.Module): The PyTorch model to apply LoRA to.
        rank (int): The rank of the LoRA update matrices (r in the LoRA paper).
                    A higher rank allows for more expressiveness but increases trainable parameters.
        lora_alpha (float): The LoRA scaling factor (alpha in the LoRA paper).
                            This value is used to scale the output of the LoRA path:
                            `scaling = lora_alpha / rank`.
        exclude_modules (Optional[list]): A list of strings. If a module's full hierarchical name
                                         (e.g., 'encoder.layer.0.attn.q_proj') contains any of
                                         these strings as a substring, that `nn.Linear` layer
                                         will be excluded from LoRA adaptation and its parameters
                                         will remain frozen along with the rest of the original model.
    """
    if exclude_modules is None:
        exclude_modules = []

    # Freeze all parameters in the entire model
    for param in model.parameters():
        param.requires_grad_(False)

    # Iterate through named modules and their direct parents
    # This approach is more robust for replacing modules in place
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_exclude = False
            for exclude_name_part in exclude_modules:
                if exclude_name_part in name:
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Get the parent module to replace the child
            # Handle top-level modules differently
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
                child_name = name

            # Replace the original nn.Linear layer with our LoRALayer
            setattr(parent_module, child_name, LoRALayer(module, rank, lora_alpha))
