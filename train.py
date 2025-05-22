import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
import random
import logging

from dataset import MMAC2023Task1Dataset, ImagePreprocessingTransform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Definition ---
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
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

# --- Training Arguments Dataclass ---
@dataclass
class TrainArgs:
    """
    Arguments for training the ResNet50 model.
    """
    images_dir: str = "1. Classification of Myopic Maculopathy/1. Images/1. Training Set"
    labels_path: str = "1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    validation_images_dir: str = "1. Classification of Myopic Maculopathy/1. Images/2. Validation Set"
    validation_labels_path: str = "1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
    
    image_size: tuple = (224, 224) # Standard input size for ResNet
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    test_split_ratio: float = 0.2 # Ratio of dataset to use for the internal test set
    eval_steps: int = 100 # Evaluate on the internal test set every N steps
    patience: int = 10 # Early stopping patience
    num_classes: int = 5 # As per the dataset description (grades 0-4)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "model_outputs"
    model_save_name: str = "resnet50_best_model.pth"
    random_seed: Optional[int] = 42 # Optional random seed for reproducibility
    pretrained: bool = True # Whether to use pretrained weights for ResNet

# --- Reproducibility Function ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} for reproducibility.")

# --- Core Training Function ---
def train(args: TrainArgs):
    """
    Trains a ResNet50 model on the specified dataset.

    Args:
        args (TrainArgs): An instance of TrainArgs containing all training parameters.
    """
    # --- Setup Reproducibility ---
    if args.random_seed is not None:
        set_seed(args.random_seed)

    # --- Setup Device and Output Directory ---
    logger.info(f"Using device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory '{args.output_dir}' ensured.")

    # --- Data Loading and Splitting ---
    logger.info("Loading full dataset...")
    transform = ImagePreprocessingTransform(output_size=args.image_size)
    full_dataset = MMAC2023Task1Dataset(
        images_dir=args.images_dir,
        labels_path=args.labels_path,
        transform=transform
    )

    # Split the main dataset into training and internal test sets
    total_size = len(full_dataset)
    test_size = int(args.test_split_ratio * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    logger.info(f"Total samples in main dataset: {total_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Internal test samples: {len(test_dataset)}")

    # Load validation dataset for final evaluation
    logger.info("Loading validation dataset...")
    validation_dataset = MMAC2023Task1Dataset(
        images_dir=args.validation_images_dir,
        labels_path=args.validation_labels_path,
        transform=transform
    )
    valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    logger.info(f"Validation samples: {len(validation_dataset)}")

    # --- Model Initialization ---
    logger.info("Initializing ResNet50Classifier model...")
    model = ResNet50Classifier(num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(args.device)
    logger.info(f"Model moved to {args.device}.")

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # --- Training Loop ---
    best_test_accuracy = 0.0
    epochs_no_improve = 0
    global_step = 0

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Use enumerate directly with DataLoader for cleaner logging
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            global_step += 1

            # --- Evaluate on internal test set ---
            if global_step % args.eval_steps == 0:
                model.eval()
                test_loss = 0.0
                correct_test = 0
                total_test = 0
                
                with torch.no_grad():
                    for test_inputs, test_labels in test_loader:
                        test_inputs, test_labels = test_inputs.to(args.device), test_labels.to(args.device)
                        test_outputs = model(test_inputs)
                        loss_test = criterion(test_outputs, test_labels)
                        
                        test_loss += loss_test.item() * test_inputs.size(0)
                        _, predicted_test = torch.max(test_outputs.data, 1)
                        total_test += test_labels.size(0)
                        correct_test += (predicted_test == test_labels).sum().item()
                
                avg_test_loss = test_loss / total_test
                test_accuracy = correct_test / total_test
                logger.info(f"--- Step {global_step} --- Internal Test Loss: {avg_test_loss:.4f}, Internal Test Accuracy: {test_accuracy:.4f}")

                # Early stopping check
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    epochs_no_improve = 0
                    # Save the best model
                    model_save_path = os.path.join(args.output_dir, args.model_save_name)
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"Saved best model to {model_save_path} with accuracy: {best_test_accuracy:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement for {epochs_no_improve} evaluations.")
                    if epochs_no_improve >= args.patience:
                        logger.info("Early stopping triggered!")
                        break # Break from inner loop
                model.train() # Switch back to train mode

        avg_train_loss = running_loss / total_train
        train_accuracy = correct_train / total_train
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} finished. Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        if epochs_no_improve >= args.patience:
            break # Break from outer loop if early stopping was triggered

    logger.info("--- Training complete. ---")

    # --- Final Evaluation on Validation Set ---
    logger.info("--- Performing final evaluation on validation set ---")
    
    # Load the best model
    best_model_path = os.path.join(args.output_dir, args.model_save_name)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path} for final validation.")
    else:
        logger.warning("No best model found, evaluating with the last trained model state.")

    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for val_inputs, val_labels in valid_loader:
            val_inputs, val_labels = val_inputs.to(args.device), val_labels.to(args.device)
            val_outputs = model(val_inputs)
            loss_valid = criterion(val_outputs, val_labels)
            
            valid_loss += loss_valid.item() * val_inputs.size(0)
            _, predicted_valid = torch.max(val_outputs.data, 1)
            total_valid += val_labels.size(0)
            correct_valid += (predicted_valid == val_labels).sum().item()

    avg_valid_loss = valid_loss / total_valid
    valid_accuracy = correct_valid / total_valid
    logger.info(f"Final Validation Loss: {avg_valid_loss:.4f}, Final Validation Accuracy: {valid_accuracy:.4f}")

if __name__ == "__main__":
    train_args = TrainArgs(
        images_dir="1. Classification of Myopic Maculopathy/1. Images/1. Training Set",
        labels_path="1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv",
        validation_images_dir="1. Classification of Myopic Maculopathy/1. Images/2. Validation Set",
        validation_labels_path="1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv",
        image_size=(224, 224),
        batch_size=32,
        num_epochs=50,
        learning_rate=0.0001,
        test_split_ratio=0.2,
        eval_steps=100,
        patience=5,
        num_classes=5,
        output_dir="model_outputs",
        model_save_name="resnet50_classifier_best.pth",
        random_seed=42, # Set a specific seed for reproducibility
        pretrained=True # Use pre-trained weights by default
    )
    
    train(train_args)