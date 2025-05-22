import os
import pandas as pd
from cv2 import imread, resize
import torch
from torch.utils.data import Dataset

class MMAC2023Task1Dataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        """
        Initializes the dataset.

        Args:
            images_dir (str): Directory containing the image files.
            labels_path (str): Path to the CSV file containing labels.
            transform (callable, optional): Optional transform to be applied on an image.
                                            If None, no transform is applied.
        """
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform # If None, no transform will be applied

        # Create a mapping from image file name to its label for efficient lookup
        self.image_to_label = pd.Series(
            self.labels_df.myopic_maculopathy_grade.values,
            index=self.labels_df.image
        ).to_dict()

        # Filter out image files that do not have a corresponding entry in the labels CSV,
        # or image files in the labels CSV that do not exist in the images directory.
        self.image_files = [
            f for f in os.listdir(images_dir)
            if f in self.image_to_label and os.path.exists(os.path.join(images_dir, f))
        ]
        
        # Sort image files to ensure consistent order
        self.image_files.sort()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: (image, label)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image using OpenCV: 1 loads a color image (BGR order)
        image = imread(img_path, 1) 
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {img_path}")

        label = self.image_to_label[img_name]
        label = torch.tensor(label, dtype=torch.long) # Convert label to a torch tensor

        if self.transform:
            image = self.transform(image)

        return image, label

class ImagePreprocessingTransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        # image is a NumPy array from cv2.imread (BGR order)
        
        image = resize(image, self.output_size)
        image = image / 255.0 # Normalize to [0, 1]
        
        # Convert NumPy array to PyTorch Tensor and permute dimensions
        # HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image = torch.from_numpy(image).permute(2, 0, 1).float() 
        
        return image

# --- Example Usage ---
if __name__ == "__main__":
    training_set_images_dir = "1. Classification of Myopic Maculopathy/1. Images/1. Training Set"
    validation_set_images_dir = "1. Classification of Myopic Maculopathy/1. Images/2. Validation Set"

    training_set_labels_path = "1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    validation_set_labels_path = "1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"

    print("--- Initializing Training Dataset ---")
    custom_transform = ImagePreprocessingTransform(output_size=(512, 512))
    train_dataset_custom = MMAC2023Task1Dataset(
        images_dir=training_set_images_dir,
        labels_path=training_set_labels_path,
        transform=custom_transform
    )
    print(f"Number of training samples: {len(train_dataset_custom)}")

    if len(train_dataset_custom) > 0:
        sample_image, sample_label = train_dataset_custom[0]
        print(f"Sample training image shape: {sample_image.shape}") # Should be [C, H, W]
        print(f"Sample training label: {sample_label.item()}")
        print(f"Sample training image dtype: {sample_image.dtype}")
        print(f"Sample training image type: {type(sample_image)}")


    print("\n--- Initializing Validation Dataset ---")
    val_dataset_no_transform = MMAC2023Task1Dataset(
        images_dir=validation_set_images_dir,
        labels_path=validation_set_labels_path,
        transform=custom_transform
    )
    print(f"Number of validation samples: {len(val_dataset_no_transform)}")

    if len(val_dataset_no_transform) > 0:
        sample_image_val, sample_label_val = val_dataset_no_transform[0]
        print(f"Sample validation image shape: {sample_image_val.shape}") # Should be [H, W, C]
        print(f"Sample validation label: {sample_label_val.item()}")
        print(f"Sample validation image dtype: {sample_image_val.dtype}")
        print(f"Sample validation image type: {type(sample_image_val)}")


    print("\n--- Testing with DataLoader for Training Set ---")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset_custom, batch_size=4, shuffle=True)
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
        print(f"Batch {batch_idx}: Images dtype {images.dtype}")
        if batch_idx == 1: # Print details for first 2 batches
            break