import os
import pandas as pd
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
    It expects HWC, BGR input and returns HWC, BGR output.
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
        Expects image in HWC, BGR order.
        Returns augmented image in HWC, BGR order.
        """
        # Albumentations expects RGB, so convert from BGR to RGB first.
        image_rgb = cvtColor(image, COLOR_BGR2RGB)
        augmented_image_rgb = self.transform(image=image_rgb)["image"]
        # Convert back to BGR as per requirement, before returning.
        augmented_image_bgr = cvtColor(augmented_image_rgb, COLOR_RGB2BGR)
        return augmented_image_bgr


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

        # Read image using OpenCV: 1 loads a color image (BGR order)
        image: np.ndarray = imread(img_path, 1)

        label: int = self.image_to_label[img_name]
        label_tensor: torch.Tensor = torch.tensor(
            label, dtype=torch.long
        )  # Convert label to a torch tensor

        for transform in self.transform:
            image = transform(image)

        image = resize(image, self.output_size)
        image = image / 255.0  # Normalize to [0, 1]
        # Convert NumPy array to PyTorch Tensor and permute dimensions
        # HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return image_tensor, label_tensor


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
        transform=[],
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
