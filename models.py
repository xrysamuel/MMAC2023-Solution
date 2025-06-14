import logging
from typing import Dict, Union, Tuple
from functools import partial

import torch
from torch import nn

from utils import apply_lora_to_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            logger.info("Initialized ResNet50 with ImageNet pre-trained weights.")
        else:
            self.resnet = resnet50(weights=None)
            logger.info("Initialized ResNet50 without pre-trained weights.")

        # Modify the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for ResNet18 with dropout rate {dropout}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        if pretrained:
            self.resnet = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )  # Use appropriate weights for ResNet18
            logger.info("Initialized ResNet18 with ImageNet pre-trained weights.")
        else:
            self.resnet = resnet18(weights=None)
            logger.info("Initialized ResNet18 without pre-trained weights.")

        # Modify the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for ResNet18 with dropout rate {dropout}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

class InceptionV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        if pretrained:
            # Inception v3 requires a specific input size (299x299) and has an auxiliary output
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True, dropout=dropout)
            logger.info("Initialized InceptionV3 with ImageNet pre-trained weights.")
        else:
            self.inception = inception_v3(weights=None, aux_logits=True, dropout=dropout)
            logger.info("Initialized InceptionV3 without pre-trained weights.")

        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)
        logger.info(
            f"Modified final layer to have {num_classes} output features for InceptionV3."
        )

        # InceptionV3 also has an auxiliary classifier ('AuxLogits.fc')
        if self.inception.AuxLogits is not None:
            num_aux_ftrs = self.inception.AuxLogits.fc.in_features
            self.inception.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_classes)
            logger.info(f"Modified auxiliary final layer to have {num_classes} output features for InceptionV3.")
        else:
            logger.warning("AuxLogits is None even though pretrained=True. This might indicate an issue.")


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.inception(x)


class EfficientNetV2SClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

        if pretrained:
            self.efficientnet = efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
            )
            logger.info(
                "Initialized EfficientNetV2-S with ImageNet pre-trained weights."
            )
        else:
            self.efficientnet = efficientnet_v2_s(weights=None)
            logger.info("Initialized EfficientNetV2-S without pre-trained weights.")

        # Modify the final classification layer (EfficientNet usually has a 'classifier' attribute)
        # It's typically a Sequential with a Linear layer as the last component.
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for EfficientNetV2-S."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


class EfficientNetV2MClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

        if pretrained:
            self.efficientnet = efficientnet_v2_m(
                weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1
            )
            logger.info(
                "Initialized EfficientNetV2-M with ImageNet pre-trained weights."
            )
        else:
            self.efficientnet = efficientnet_v2_m(weights=None)
            logger.info("Initialized EfficientNetV2-M without pre-trained weights.")

        # Modify the final classification layer (EfficientNet usually has a 'classifier' attribute)
        # It's typically a Sequential with a Linear layer as the last component.
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for EfficientNetV2-S."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

        if pretrained:
            self.mobilenet = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            logger.info(
                "Initialized MobileNetV3-Large with ImageNet pre-trained weights."
            )
        else:
            self.mobilenet = mobilenet_v3_large(weights=None)
            logger.info("Initialized MobileNetV3-Large without pre-trained weights.")

        # Modify the final classification layer
        # MobileNetV3's classifier is usually a Sequential, with the last layer being the Linear layer.
        num_ftrs = self.mobilenet.classifier[
            3
        ].in_features  # Check the exact index for your MobileNetV3 version
        self.mobilenet.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for MobileNetV3-Large."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(x)


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
    
class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5, model_size: str = 'tiny'):
        super().__init__()
        from torchvision.models import convnext_base, convnext_small, convnext_tiny, ConvNeXt_Base_Weights, ConvNeXt_Small_Weights, ConvNeXt_Tiny_Weights

        if model_size == 'base':
            if pretrained:
                self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                logger.info("Initialized ConvNeXt-Base with ImageNet pre-trained weights.")
            else:
                self.convnext = convnext_base(weights=None)
                logger.info("Initialized ConvNeXt-Base without pre-trained weights.")
        elif model_size == 'small':
            if pretrained:
                self.convnext = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
                logger.info("Initialized ConvNeXt-Small with ImageNet pre-trained weights.")
            else:
                self.convnext = convnext_small(weights=None)
                logger.info("Initialized ConvNeXt-Small without pre-trained weights.")
        elif model_size == 'tiny':
            if pretrained:
                self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                logger.info("Initialized ConvNeXt-Tiny with ImageNet pre-trained weights.")
            else:
                self.convnext = convnext_tiny(weights=None)
                logger.info("Initialized ConvNeXt-Tiny without pre-trained weights.")
        else:
            raise ValueError(f"Unsupported ConvNeXt model_size: {model_size}. Choose from 'tiny', 'small', 'base'.")

        # ConvNeXt models have a different head structure.
        # The classifier is usually convnext.classifier[2] (Linear layer)
        # We need to replace the entire classifier module or just its last linear layer.
        # torchvision's ConvNeXt models have `self.classifier` which is `nn.Sequential`
        # The last layer is usually a `nn.Linear` at index 2 or 3 depending on batch norm presence.
        num_ftrs = self.convnext.classifier[-1].in_features
        self.convnext.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for ConvNeXt-{model_size} with dropout rate {dropout}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)


class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5, model_size: str = 'tiny'):
        super().__init__()
        from torchvision.models import swin_b, swin_s, swin_t, Swin_B_Weights, Swin_S_Weights, Swin_T_Weights

        if model_size == 'base':
            if pretrained:
                self.swin_transformer = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
                logger.info("Initialized SwinTransformer-Base with ImageNet pre-trained weights.")
            else:
                self.swin_transformer = swin_b(weights=None)
                logger.info("Initialized SwinTransformer-Base without pre-trained weights.")
        elif model_size == 'small':
            if pretrained:
                self.swin_transformer = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
                logger.info("Initialized SwinTransformer-Small with ImageNet pre-trained weights.")
            else:
                self.swin_transformer = swin_s(weights=None)
                logger.info("Initialized SwinTransformer-Small without pre-trained weights.")
        elif model_size == 'tiny':
            if pretrained:
                self.swin_transformer = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
                logger.info("Initialized SwinTransformer-Tiny with ImageNet pre-trained weights.")
            else:
                self.swin_transformer = swin_t(weights=None)
                logger.info("Initialized SwinTransformer-Tiny without pre-trained weights.")
        else:
            raise ValueError(f"Unsupported SwinTransformer model_size: {model_size}. Choose from 'tiny', 'small', 'base'.")

        # SwinTransformer models in torchvision have a `head` attribute which is the final Linear layer.
        num_ftrs = self.swin_transformer.head.in_features
        self.swin_transformer.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        logger.info(
            f"Modified final layer to have {num_classes} output features for SwinTransformer-{model_size} with dropout rate {dropout}."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swin_transformer(x)

MODEL_CLASS_DICT: Dict[str, nn.Module] = {
    "resnet50": ResNet50Classifier,
    "mobilenetv3s": MobileNetV3Classifier,
    "resnet18": ResNet18Classifier,
    "effnetv2s": EfficientNetV2SClassifier,
    "effnetv2m": EfficientNetV2MClassifier,
    "retfound": RETFoundMAEClassifier,
    "inceptionv3": InceptionV3Classifier,
    "convnext": ConvNeXtClassifier,
    "swintransformer": SwinTransformerClassifier
}
