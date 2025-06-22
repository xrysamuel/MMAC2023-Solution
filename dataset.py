import os

import pandas as pd
import cv2
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR, BORDER_CONSTANT
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Callable, Optional, Dict, List, Union
import albumentations as alb
import matplotlib.pyplot as plt

class Transform:
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        return image
    
    def transform_encoded_label(self, one_hot: torch.Tensor) -> torch.Tensor:
        return one_hot


class MMAC2023Task1Dataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_path: str,
        transform: List[Transform],
        output_size: Tuple[int, int] = (224, 224),
        categories: int = 5
    ):
        """
        Initializes the dataset.

        Args:
            images_dir (str): Directory containing the image files.
            labels_path (str): Path to the CSV file containing labels.
            transform (list[Transform]): Transform to be applied on an image.
        """
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.output_size = output_size
        self.categories = categories

        # Create a mapping from image file name to its label for efficient lookup
        self.image_to_label: Dict[str, int] = pd.Series(
            self.labels_df.myopic_maculopathy_grade.values, index=self.labels_df.image
        ).to_dict()

        # Filter out image files that do not have a corresponding entry in the labels CSV,
        # or image files in the labels CSV that do not exist in the images directory.
        self.image_files: List[str] = [
            f
            for f in os.listdir(images_dir)
            if f in self.image_to_label and os.path.exists(os.path.join(images_dir, f))
        ]

        # Sort image files to ensure consistent order
        self.image_files.sort()

        # Define ImageNet standard mean and standard deviation as tensors
        # These are for channels in RGB order
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (image_tensor, original_label_tensor, one_hot_mixed_label_tensor)
            where image_tensor is the processed image, original_label_tensor is the original integer label,
            and one_hot_mixed_label_tensor is the (potentially mixed) one-hot encoded label.
        """
        
        img_name, image = self.get_image(idx)

        label_tensor, encoded_label = self.get_label(img_name)

        for transform in self.transform:
            image = transform.transform_image(image)
            encoded_label = transform.transform_encoded_label(encoded_label)

        image_tensor = self.final_preprocess(image)

        return image_tensor, label_tensor, encoded_label
    
    def final_preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Applies final preprocessing steps to an image: resizing, normalization,
        and conversion to PyTorch tensor with channel-first order.

        Args:
            image (np.ndarray): Input image in HWC format, 0-255 range.

        Returns:
            torch.Tensor: Processed image tensor in CHW format, normalized.
        """
        output_image = resize(image, self.output_size)
        output_image = output_image / 255.0  # Normalize to [0, 1]

        # Convert NumPy array to PyTorch Tensor and permute dimensions
        # HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image_tensor = torch.from_numpy(output_image).permute(2, 0, 1).float()

        # Apply ImageNet standard normalization (mean and std)
        image_tensor = (image_tensor - self.imagenet_mean) / self.imagenet_std
        return image_tensor
    
    def get_image(self, idx: int) -> Tuple[str, np.ndarray]:
        """
        Retrieves an image by index.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            Tuple[str, np.ndarray]: A tuple containing the image file name and the image as a NumPy array (RGB).
        """
        img_name: str = self.image_files[idx]
        img_path: str = os.path.join(self.images_dir, img_name)

        # Read image using OpenCV: 1 loads a color image (BGR order)
        image: np.ndarray = imread(img_path, 1)
        image = cvtColor(image, COLOR_BGR2RGB)  # convert to RGB
        return img_name, image
    
    def get_label(self, img_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the label for a given image name.

        Args:
            img_name (str): The name of the image file.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the integer label as a long tensor
            and the one-hot encoded label as a float tensor.
        """
        label: int = self.image_to_label[img_name]
        label_tensor = torch.tensor(label, dtype=torch.long)
        encoded_label = torch.zeros(self.categories, dtype=torch.float32)
        encoded_label[label] = 1.0
        return label_tensor, encoded_label
    
    def get_image_label_pair_df(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame with image names and their corresponding labels.

        Returns:
            pd.DataFrame: A DataFrame with two columns: 'image' (str) and 'label' (int).
        """
        image_paths = [os.path.join(self.images_dir, name) for name in self.image_files]
        labels = [self.image_to_label[img_name] for img_name in self.image_files]

        df = pd.DataFrame({
            'image': image_paths,
            'label': labels
        })
        return df


class AugmentationTransform(Transform):
    """
    Encapsulates various data augmentation techniques using Albumentations.
    It expects HWC, RGB input and returns HWC, RGB output.
    """
    def __init__(self):
        """
        Initializes the augmentation pipeline.
        """
        self.transform = alb.Compose(
            [
                # Geometric Transformations
                alb.HorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
                # Rotate with small random angles, filling exposed areas with black
                # border_mode=cv2.BORDER_CONSTANT and value=0 ensures black padding.
                alb.Rotate(limit=10, p=0.8, border_mode=BORDER_CONSTANT),
                # Pixel-Value Transformations
                alb.GaussianBlur(blur_limit=(1, 3), p=0.5),  # Apply Gaussian blur
                alb.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.6
                ),  # Adjust brightness/contrast
                alb.HueSaturationValue(
                    hue_shift_limit=4, sat_shift_limit=20, val_shift_limit=20, p=0.6
                ),  # Adjust hue, saturation, value
            ]
        )

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the defined Albumentations augmentations.
        Expects image in HWC, RGB order.
        Returns augmented image in HWC, RGB order.
        """
        augmented_image = self.transform(image=image)["image"]
        return augmented_image

class RandomCropTransform(Transform):
    def __init__(self, size: int = 224, scale_size: int = 256):
        """
        Initializes the random crop transformation.
        Args:
            size (int): The final square side length of the cropped image, default is 224.
            scale_size (int): The size to which the shorter side of the image will be scaled
                              before cropping, default is 256.
        """
        self.size = size
        self.scale_size = scale_size

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the random crop transformation to an image.
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

        # 2. Randomly crop to self.size x self.size.
        h_resized, w_resized, _ = image.shape
        # Ensure crop region doesn't go out of bounds
        if h_resized > self.size:
            top = np.random.randint(0, h_resized - self.size + 1)
        else:
            top = 0 # If the image itself is smaller than target size, crop from top
        if w_resized > self.size:
            left = np.random.randint(0, w_resized - self.size + 1)
        else:
            left = 0 # If the image itself is smaller than target size, crop from left
        image = image[top:top + self.size, left:left + self.size, :]
        return image

class CenterCropTransform(Transform):
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

    def transform_image(self, image: np.ndarray) -> np.ndarray:
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

class CutMixTransform(Transform):
    """
    Implements the CutMix data augmentation technique.
    It cuts a patch from one image and pastes it onto another, then mixes their labels
    proportionally to the area of the pasted patch.
    """
    def __init__(self, ref_dataset: 'MMAC2023Task1Dataset', rate: float = 0.3):
        """
        Initializes the CutMix transformation.
        Args:
            ref_dataset (MMAC2023Task1Dataset): The dataset instance from which to sample
                                                another image for the CutMix operation.
            rate (float): The proportion of the patch size relative to the original image size.
                          Must be between 0.0 and 0.7 (exclusive).
        """
        assert 0.0 < rate and rate < 0.7, "Rate must be between 0.0 and 0.7 (exclusive)"
        self.rate = rate
        self.ref_dataset = ref_dataset
        self.lambda_value: float = 1.0 # This will be calculated in transform_image
        self.another_one_hot_label: Optional[torch.Tensor] = None
    
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the CutMix transformation to the image.
        Cuts a patch from a randomly selected image from the dataset and pastes it onto the current image.
        Args:
            image (np.ndarray): Input image in HWC format, 0-255 range.
        Returns:
            np.ndarray: Transformed image with a patch from another image.
        """
        if self.ref_dataset is None:
            raise ValueError("ref_dataset must be set for CutMixTransform. It should be passed during initialization.")

        h, w, c = image.shape

        # Randomly select another image from the dataset
        random_idx = np.random.randint(0, len(self.ref_dataset))
        # We need to call get_image and get_label from the dataset instance
        _, another_image = self.ref_dataset.get_image(random_idx)
        _, self.another_one_hot_label = self.ref_dataset.get_label(self.ref_dataset.image_files[random_idx])

        # Resize another_image to the same size as the current image
        another_image = cv2.resize(another_image, (w, h), interpolation=cv2.INTER_AREA)

        # Calculate patch dimensions based on rate
        patch_h = int(h * self.rate)
        patch_w = int(w * self.rate)

        # Ensure patch dimensions are at least 1
        patch_h = max(1, patch_h)
        patch_w = max(1, patch_w)

        # Randomly choose top-left corner of the patch for both images
        x1 = np.random.randint(0, w - patch_w + 1)
        y1 = np.random.randint(0, h - patch_h + 1)
        
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        # Calculate the area of the pasted patch relative to the total image area
        area_patch = patch_w * patch_h
        area_total = w * h
        self.lambda_value = 1.0 - (area_patch / area_total)

        # Apply the CutMix: paste the patch from another_image onto the current image
        mixed_image = np.copy(image)
        mixed_image[y1:y2, x1:x2, :] = another_image[y1:y2, x1:x2, :]

        return mixed_image
    
    def transform_encoded_label(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Applies the CutMix transformation to the one-hot encoded label.
        Mixes the current label with the label of the pasted image based on the lambda value.
        Args:
            one_hot (torch.Tensor): The original one-hot encoded label tensor.
        Returns:
            torch.Tensor: The mixed one-hot encoded label tensor.
        """
        if self.another_one_hot_label is None:
            # If transform_image was not called or failed, return original label
            return one_hot 

        mixed_one_hot_label = self.lambda_value * one_hot + (1.0 - self.lambda_value) * self.another_one_hot_label
        return mixed_one_hot_label


class Visualization:
    """
    A utility class to help visualize image transformations in a pipeline.
    It includes nested Transform classes to collect, draw, and discard images.
    """
    def __init__(self):
        self.images = []

    class Collect(Transform):
        def __init__(self, vis: "Visualization"):
            self.vis = vis

        def transform_image(self, image):
            self.vis.images.append(image)
            return image

    class Draw(Transform):
        def __init__(self, vis: "Visualization"):
            self.vis = vis

        def transform_image(self, image):
            if not self.vis.images:
                print("No images to draw. Please use the 'collect' method first.")
                return

            num_images = len(self.vis.images)

            # Create a figure and a set of subplots, all in one row
            # figsize adjusted to accommodate all images side-by-side
            fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))

            # Ensure axes is an iterable array even for a single image
            if num_images == 1:
                axes = [axes]

            for idx, image in enumerate(self.vis.images):
                ax = axes[idx]
                ax.imshow(image)
                ax.axis("off")  # Turn off axis labels and ticks
                ax.set_title(f"Image {idx+1}")  # Optionally add a title to each image

            plt.tight_layout()  # Adjust subplot parameters for a tight layout
            plt.show()  # Display the plot

            return image

    class Discard(Transform):
        def __init__(self, vis: "Visualization"):
            self.vis = vis

        def transform_image(self, image):
            self.vis.images.clear()
            return image


# --- Example Usage ---
if __name__ == "__main__":
    from data_utils import image_dataset_analyze
    training_set_images_dir: str = (
        "1. Classification of Myopic Maculopathy/1. Images/1. Training Set"
    )
    validation_set_images_dir: str = (
        "1. Classification of Myopic Maculopathy/1. Images/2. Validation Set"
    )

    training_set_labels_path: str = (
        "1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    )
    validation_set_labels_path: str = (
        "1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
    )

    aug_transform = AugmentationTransform()
    visualization = Visualization()

    print("--- Initializing Training Dataset ---")
    ref_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
        images_dir=training_set_images_dir,
        labels_path=training_set_labels_path,
        transform=[]
    )
    train_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
        images_dir=training_set_images_dir,
        labels_path=training_set_labels_path,
        transform=[
            Visualization.Collect(visualization),
            CutMixTransform(ref_dataset),
            AugmentationTransform(),
            RandomCropTransform(),
            Visualization.Collect(visualization),
            Visualization.Draw(visualization),
            Visualization.Discard(visualization),
        ],
    )
    print(f"Number of training samples: {len(train_dataset)}")

    if len(train_dataset) > 0:
        sample_image, sample_label, sample_encoded_label = train_dataset[0]
        print(
            f"Sample training image shape: {sample_image.shape}"
        )  # Should be [C, H, W]
        print(f"Sample training label: {sample_label.item()}")
        print(f"Sample training encoded label: {sample_encoded_label}")
        print(f"Sample training image dtype: {sample_image.dtype}")
        print(f"Sample training image type: {type(sample_image)}")

    print("\n--- Initializing Validation Dataset ---")
    val_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
        images_dir=validation_set_images_dir,
        labels_path=validation_set_labels_path,
        transform=[CenterCropTransform()],
    )
    print(f"Number of validation samples: {len(val_dataset)}")

    if len(val_dataset) > 0:
        sample_image_val, sample_label_val, sample_encoded_label_val = val_dataset[0]
        print(
            f"Sample validation image shape: {sample_image_val.shape}"
        )  # Should be [C, H, W] after transform
        print(f"Sample validation label: {sample_label_val.item()}")
        print(f"Sample validation encoded label: {sample_encoded_label_val}")
        print(f"Sample validation image dtype: {sample_image_val.dtype}")
        print(f"Sample validation image type: {type(sample_image_val)}")

    print("\n--- Testing with DataLoader for Training Set ---")
    from torch.utils.data import DataLoader

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_idx, (images, labels, encoded_labels) in enumerate(train_loader):
        print(
            f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}, Labels {labels}"
        )
        print(f"Batch {batch_idx}: Images dtype {images.dtype}")
        break

    # image_dataset_analyze(train_dataset.get_image_label_pair_df(), "output/data/train")
    # image_dataset_analyze(val_dataset.get_image_label_pair_df(), "output/data/valid")

    