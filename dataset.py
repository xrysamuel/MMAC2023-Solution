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


class AugmentationTransform:
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

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the defined Albumentations augmentations.
        Expects image in HWC, RGB order.
        Returns augmented image in HWC, RGB order.
        """
        augmented_image = self.transform(image=image)["image"]
        return augmented_image

class RandomCropTransform:
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

    def __call__(self, image: np.ndarray) -> np.ndarray:
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

class MMAC2023Task1Dataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_path: str,
        transform: List[Callable[[np.ndarray], np.ndarray]],
        output_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initializes the dataset.

        Args:
            images_dir (str): Directory containing the image files.
            labels_path (str): Path to the CSV file containing labels.
            transform (list[callable]): Transform to be applied on an image.
        """
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.output_size = output_size

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, label) where image is a torch.Tensor and label is a torch.Tensor.
        """
        img_name: str = self.image_files[idx]
        img_path: str = os.path.join(self.images_dir, img_name)

        label: int = self.image_to_label[img_name]
        label_tensor: torch.Tensor = torch.tensor(
            label, dtype=torch.long
        )  # Convert label to a torch tensor

        # Read image using OpenCV: 1 loads a color image (BGR order)
        image: np.ndarray = imread(img_path, 1)
        image = cvtColor(image, COLOR_BGR2RGB)  # convert to RGB

        for transform in self.transform:
            image = transform(image)

        image = resize(image, self.output_size)
        image = image / 255.0  # Normalize to [0, 1]

        # Convert NumPy array to PyTorch Tensor and permute dimensions
        # HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # Apply ImageNet standard normalization (mean and std)
        image_tensor = (image_tensor - self.imagenet_mean) / self.imagenet_std

        return image_tensor, label_tensor
    
    def get_image_label_pair_df(self) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame with image names and their corresponding labels.

        Returns:
            pd.DataFrame: A DataFrame with two columns: 'image' (str) and 'label' (int).
        """
        image_paths = [os.path.join(self.images_dir, name) for name in self.image_names]
        labels = [self.image_to_label[img_name] for img_name in self.image_files]

        df = pd.DataFrame({
            'image': image_paths,
            'label': labels
        })
        return df


class Visualization:
    def __init__(self):
        self.images = []

    def collect(self, image):
        self.images.append(image)
        return image

    def draw(self, in_image):
        if not self.images:
            print("No images to draw. Please use the 'collect' method first.")
            return

        num_images = len(self.images)

        # Create a figure and a set of subplots, all in one row
        # figsize adjusted to accommodate all images side-by-side
        fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))

        # Ensure axes is an iterable array even for a single image
        if num_images == 1:
            axes = [axes]

        for idx, image in enumerate(self.images):
            ax = axes[idx]
            ax.imshow(image)
            ax.axis("off")  # Turn off axis labels and ticks
            ax.set_title(f"Image {idx+1}")  # Optionally add a title to each image

        plt.tight_layout()  # Adjust subplot parameters for a tight layout
        plt.show()  # Display the plot

        return in_image

    def discard_all(self, in_image):
        self.images.clear()
        return in_image


# --- Example Usage ---
if __name__ == "__main__":
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
    vis_transform = Visualization()

    print("--- Initializing Training Dataset ---")
    train_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
        images_dir=training_set_images_dir,
        labels_path=training_set_labels_path,
        transform=[
            vis_transform.collect,
            AugmentationTransform(),
            RandomCropTransform(),
            vis_transform.collect,
            vis_transform.draw,
            vis_transform.discard_all,
        ],
    )
    print(f"Number of training samples: {len(train_dataset)}")

    if len(train_dataset) > 0:
        sample_image, sample_label = train_dataset[0]
        print(
            f"Sample training image shape: {sample_image.shape}"
        )  # Should be [C, H, W]
        print(f"Sample training label: {sample_label.item()}")
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
        sample_image_val, sample_label_val = val_dataset[0]
        print(
            f"Sample validation image shape: {sample_image_val.shape}"
        )  # Should be [C, H, W] after transform
        print(f"Sample validation label: {sample_label_val.item()}")
        print(f"Sample validation image dtype: {sample_image_val.dtype}")
        print(f"Sample validation image type: {type(sample_image_val)}")

    print("\n--- Testing with DataLoader for Training Set ---")
    from torch.utils.data import DataLoader

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(
            f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}, Labels {labels}"
        )
        print(f"Batch {batch_idx}: Images dtype {images.dtype}")
        if batch_idx == 2:  # Print details for first 3 batches
            break
