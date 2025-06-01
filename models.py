import torch
import torch.nn as nn
import logging
from typing import Dict
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
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
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        logger.info(f"Modified final layer to have {num_classes} output features.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
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
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        logger.info(
            f"Modified final layer to have {num_classes} output features for ResNet18."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class EfficientNetV2SClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
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
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
        logger.info(
            f"Modified final layer to have {num_classes} output features for EfficientNetV2-S."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


class EfficientNetV2MClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
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
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
        logger.info(
            f"Modified final layer to have {num_classes} output features for EfficientNetV2-S."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)


class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
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
        self.mobilenet.classifier[3] = nn.Linear(num_ftrs, num_classes)
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

    def __init__(self, num_classes: int = 5, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        from timm.models.layers import trunc_normal_

        if pretrained:
            self.retfoundmae = self.VisionTransformer(num_classes)
            logger.info("Attempting to download model from Hugging Face Hub.")
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=f'YukunZhou/RETFound_mae_natureCFP',
                filename=f'RETFound_mae_natureCFP.pth',
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
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    logger.info(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            trunc_normal_(self.retfoundmae.head.weight, std=2e-5)
            logger.info("Initialized RETFoundMAE without pre-trained weights.")

        if freeze_backbone and pretrained:
            for name, param in self.retfoundmae.named_parameters():
                if "head" not in name:
                    param.requires_grad = False
                else:
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

MODEL_CLASS_DICT: Dict[str, nn.Module] = {
    "resnet50": ResNet50Classifier,
    "mobilenetv3s": MobileNetV3Classifier,
    "resnet18": ResNet18Classifier,
    "effnetv2s": EfficientNetV2SClassifier,
    "effnetv2m": EfficientNetV2MClassifier,
    "retfound": RETFoundMAEClassifier,
}
