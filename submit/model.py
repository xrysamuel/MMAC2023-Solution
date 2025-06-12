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
        self.checkpoint = "resnet18_best.pth"
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
        self.model = ResNet18Classifier(num_classes=5, pretrained=False)
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


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # Use appropriate weights for ResNet18
        else:
            self.resnet = resnet18(weights=None)
        
        # Modify the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
