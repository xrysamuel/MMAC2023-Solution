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
        self.checkpoint = "convnext_best.pth"
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
        self.model = ConvNeXtClassifier(num_classes=5, pretrained=False)
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