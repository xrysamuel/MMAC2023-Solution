from dataclasses import dataclass, field
from typing import Optional, Tuple
import os
import sys
import numpy as np
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import HfArgumentParser

from dataset import MMAC2023Task1Dataset, AugmentationTransform, CenterCropTransform, RandomCropTransform
from models import MODEL_CLASS_DICT
from utils import temp_eval, get_model_params, get_module_param_names

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Training Arguments Dataclass
@dataclass
class TrainArgs:
    """
    Arguments for training.
    """
    images_dir: str = "1. Classification of Myopic Maculopathy/1. Images/1. Training Set"
    labels_path: str = "1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    validation_images_dir: str = "1. Classification of Myopic Maculopathy/1. Images/2. Validation Set"
    validation_labels_path: str = "1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
    
    image_size: Tuple[int, int] = (224, 224) # Standard input size for ResNet
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    test_split_ratio: float = 0.2 # Ratio of dataset to use for the internal test set
    eval_steps: int = 100 # Evaluate on the internal test set every N steps
    patience: int = 10 # Early stopping patience
    num_classes: int = 5 # As per the dataset description (grades 0-4)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    output_dir: str = "model_outputs"
    model_name: str = "resnet50"
    model_save_name: str = "resnet50_best_model.pth"
    random_seed: Optional[int] = 42 # Optional random seed for reproducibility
    pretrained: bool = True # Whether to use pretrained weights
    train_test_split: bool = False

    label_smoothing: float = 0.0

# Reproducibility Function
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} for reproducibility.")


class Trainer:
    def __init__(self, args: TrainArgs):
        self.args = args

        # Setup Reproducibility
        if args.random_seed is not None:
            set_seed(args.random_seed)

        # Setup Device and Output Directory
        logger.info(f"Using device: {args.device}")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory '{args.output_dir}' ensured.")


    def _get_dataloader(self):
        args = self.args

        logger.info("Loading full dataset...")
        full_dataset = MMAC2023Task1Dataset(
            images_dir=args.images_dir,
            labels_path=args.labels_path,
            transform=[AugmentationTransform(), RandomCropTransform()]
        )

        # Load validation dataset for final evaluation
        logger.info("Loading validation dataset...")
        validation_dataset = MMAC2023Task1Dataset(
            images_dir=args.validation_images_dir,
            labels_path=args.validation_labels_path,
            transform=[CenterCropTransform()]
        )

        if args.train_test_split:
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


            valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
            logger.info(f"Validation samples: {len(validation_dataset)}")

            return train_loader, test_loader, valid_loader
        else:
            total_size = len(full_dataset)

            train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
            logger.info(f"Training samples: {total_size}")

            valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
            logger.info(f"Validation samples: {len(validation_dataset)}")

            return train_loader, valid_loader, valid_loader

    
    def _get_model(self):
        args = self.args
        logger.info("Initializing model...")
        Model = MODEL_CLASS_DICT[args.model_name]
        model = Model(num_classes=args.num_classes, pretrained=args.pretrained)
        model.to(args.device)
        logger.info(get_model_params(model))
        logger.info(get_module_param_names(model))
        logger.info(f"Model moved to {args.device}.")
        return model
    
    def _get_criterion(self):
        args = self.args
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    def _get_optimizer(self, model: nn.Module):
        args = self.args
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        return optimizer
    
    def _evaluation(self, dataloader, model, criterion):
        args = self.args

        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                loss_test = criterion(outputs, labels)
                
                loss += loss_test.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / total
        accuracy = correct / total
        return avg_loss, accuracy


    def train(self):
        train_loader, test_loader, valid_loader = self._get_dataloader()
        model = self._get_model()
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(model)
        args = self.args

        best_test_accuracy = 0.0
        epochs_no_improve = 0
        global_step = 0

        logger.info("Starting training...")
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

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

                # Evaluate on internal test set
                if global_step % args.eval_steps == 0:
                    with temp_eval(model) as eval_model:
                        avg_test_loss, test_accuracy = self._evaluation(test_loader, eval_model, criterion)
                    logger.info(f"Step {global_step}. Internal Test Loss: {avg_test_loss:.4f}, Internal Test Accuracy: {test_accuracy:.4f}")

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

            avg_train_loss = running_loss / total_train
            train_accuracy = correct_train / total_train
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} finished. Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            if epochs_no_improve >= args.patience:
                break # Break from outer loop if early stopping was triggered

        logger.info("Training complete.")

        # Final Evaluation on Validation Set
        logger.info("Performing final evaluation on validation set.")
        best_model_path = os.path.join(args.output_dir, args.model_save_name)
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model from {best_model_path} for final validation.")
        else:
            logger.warning("No best model found, evaluating with the last trained model state.")

        with temp_eval(model) as eval_model:
            avg_valid_loss, valid_accuracy = self._evaluation(valid_loader, eval_model, criterion)
        logger.info(f"Final Validation Loss: {avg_valid_loss:.4f}, Final Validation Accuracy: {valid_accuracy:.4f}")


if __name__ == "__main__":
    yaml_file = sys.argv[1]

    parser = HfArgumentParser((TrainArgs))

    try:
        train_args, = parser.parse_yaml_file(yaml_file=yaml_file)
    except Exception as e:
        logging.error(f"Error parsing YAML file: {e}")
        exit()

    logging.info(train_args)
    
    trainer = Trainer(train_args)
    trainer.train()