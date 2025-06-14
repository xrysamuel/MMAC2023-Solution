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

from dataset import (
    MMAC2023Task1Dataset,
    AugmentationTransform,
    CenterCropTransform,
    RandomCropTransform,
)
from models import MODEL_CLASS_DICT
from utils import temp_eval, get_model_params, get_module_param_names

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Training Arguments Dataclass
@dataclass
class TrainArgs:
    """
    Arguments for training.
    """

    images_dir: str = (
        "1. Classification of Myopic Maculopathy/1. Images/1. Training Set"
    )
    labels_path: str = (
        "1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv"
    )
    validation_images_dir: str = (
        "1. Classification of Myopic Maculopathy/1. Images/2. Validation Set"
    )
    validation_labels_path: str = (
        "1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv"
    )

    image_size: Tuple[int, int] = (224, 224)  # Standard input size for ResNet
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    test_split_ratio: float = 0.2  # Ratio of dataset to use for the internal test set
    eval_steps: int = 100  # Evaluate on the internal test set every N steps
    patience: int = 10  # Early stopping patience
    num_classes: int = 5  # As per the dataset description (grades 0-4)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_dir: str = "model_outputs"
    model_name: str = "resnet50"
    model_save_name: str = "resnet50_best_model.pth"
    random_seed: Optional[int] = 42  # Optional random seed for reproducibility
    pretrained: bool = True  # Whether to use pretrained weights
    train_test_split: bool = False

    label_smoothing: float = 0.0

    aux_logits_loss_weight: float = 0.4  # for Inception series

    model_kwargs: dict = field(default_factory=dict)

    tent: bool = False
    tent_learning_rate: float = 1e-2  # Learning rate for TENT
    tent_steps: int = 3  # Number of forward passes to update BN stats
    tent_epochs: int = 1  # Number of times to iterate through the test dataset


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
            transform=[AugmentationTransform(), RandomCropTransform()],
            output_size=args.image_size,
        )

        # Load validation dataset for final evaluation
        logger.info("Loading validation dataset...")
        validation_dataset = MMAC2023Task1Dataset(
            images_dir=args.validation_images_dir,
            labels_path=args.validation_labels_path,
            transform=[CenterCropTransform()],
            output_size=args.image_size,
        )

        if args.train_test_split:
            # Split the main dataset into training and internal test sets
            total_size = len(full_dataset)
            test_size = int(args.test_split_ratio * total_size)
            train_size = total_size - test_size
            train_dataset, test_dataset = random_split(
                full_dataset, [train_size, test_size]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=os.cpu_count() // 2 or 1,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=os.cpu_count() // 2 or 1,
            )

            logger.info(f"Total samples in main dataset: {total_size}")
            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Internal test samples: {len(test_dataset)}")

            valid_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=os.cpu_count() // 2 or 1,
            )
            logger.info(f"Validation samples: {len(validation_dataset)}")

            return train_loader, test_loader, valid_loader
        else:
            total_size = len(full_dataset)

            train_loader = DataLoader(
                full_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=os.cpu_count() // 2 or 1,
            )
            logger.info(f"Training samples: {total_size}")

            valid_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=os.cpu_count() // 2 or 1,
            )
            logger.info(f"Validation samples: {len(validation_dataset)}")

            return train_loader, valid_loader, valid_loader

    def _get_model(self):
        args = self.args
        logger.info("Initializing model...")
        Model = MODEL_CLASS_DICT[args.model_name]
        model = Model(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            **args.model_kwargs,
        )
        model.to(args.device)
        logger.info(get_model_params(model))
        logger.info(get_module_param_names(model))
        logger.info(f"Model moved to {args.device}.")
        return model

    def _get_model_no_logging(self):
        args = self.args
        logger.info("Initializing model...")
        Model = MODEL_CLASS_DICT[args.model_name]
        model = Model(num_classes=args.num_classes, pretrained=args.pretrained)
        model.to(args.device)
        logger.info(f"Model moved to {args.device}.")
        return model

    def _criterion(self, outputs, labels):
        args = self.args
        criterion_base = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            main_logits = outputs[0]
            aux_logits = outputs[1]
            main_loss = criterion_base(main_logits, labels)
            aux_loss = criterion_base(aux_logits, labels)
            total_loss = main_loss + aux_loss * args.aux_logits_loss_weight
        elif isinstance(outputs, torch.Tensor):
            total_loss = criterion_base(outputs, labels)
        else:
            raise TypeError(
                f"Unsupported outputs type: {type(outputs)}. Expected torch.Tensor or a tuple containing logits."
            )

        return total_loss

    def _get_predicted_labels(self, outputs) -> torch.Tensor:
        if isinstance(outputs, tuple) and len(outputs) == 2:
            main_logits = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            main_logits = outputs
        else:
            raise TypeError(
                f"Unsupported outputs type: {type(outputs)}. Expected torch.Tensor or a tuple containing logits."
            )

        _, predicted_labels = torch.max(main_logits.data, 1)

        return predicted_labels

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
        train_loader, test_loader, valid_loader, model, criterion, optimizer = (
            self._before_train()
        )
        self._train(train_loader, test_loader, model, criterion, optimizer)
        self._after_train(model, valid_loader, criterion)

    def _before_train(self):
        train_loader, test_loader, valid_loader = self._get_dataloader()
        model = self._get_model()
        criterion = self._criterion
        optimizer = self._get_optimizer(model)
        return train_loader, test_loader, valid_loader, model, criterion, optimizer

    def _train(self, train_loader, test_loader, model, criterion, optimizer):
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
                predicted = self._get_predicted_labels(outputs)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                global_step += 1

                # Evaluate on internal test set
                if global_step % args.eval_steps == 0:
                    with temp_eval(model) as eval_model:
                        avg_test_loss, test_accuracy = self._evaluation(
                            test_loader, eval_model, criterion
                        )
                    logger.info(
                        f"Step {global_step}. Internal Test Loss: {avg_test_loss:.4f}, Internal Test Accuracy: {test_accuracy:.4f}"
                    )

                    # Early stopping check
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        epochs_no_improve = 0
                        # Save the best model
                        model_save_path = os.path.join(
                            args.output_dir, args.model_save_name
                        )
                        torch.save(model.state_dict(), model_save_path)
                        logger.info(
                            f"Saved best model to {model_save_path} with accuracy: {best_test_accuracy:.4f}"
                        )
                    else:
                        epochs_no_improve += 1
                        logger.info(
                            f"No improvement for {epochs_no_improve} evaluations."
                        )
                        if epochs_no_improve >= args.patience:
                            logger.info("Early stopping triggered!")
                            break  # Break from inner loop

            avg_train_loss = running_loss / total_train
            train_accuracy = correct_train / total_train
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs} finished. Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )

            if epochs_no_improve >= args.patience:
                break  # Break from outer loop if early stopping was triggered

        logger.info("Training complete.")

    def _after_train(self, model: nn.Module, valid_loader, criterion):
        args = self.args

        # Final Evaluation on Validation Set
        logger.info("Performing final evaluation on validation set.")
        best_model_path = os.path.join(args.output_dir, args.model_save_name)
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            logger.info(
                f"Loaded best model from {best_model_path} for final validation."
            )
        else:
            logger.warning(
                "No best model found, evaluating with the last trained model state."
            )

        with temp_eval(model) as eval_model:
            avg_valid_loss, valid_accuracy = self._evaluation(
                valid_loader, eval_model, criterion
            )
        logger.info(
            f"Final Validation Loss: {avg_valid_loss:.4f}, Final Validation Accuracy: {valid_accuracy:.4f}"
        )

        if args.tent:
            self._tent(model)
            with temp_eval(model) as eval_model:
                avg_valid_loss, valid_accuracy = self._evaluation(
                    valid_loader, eval_model, criterion
                )
            logger.info(
                f"After TENT, Final Validation Loss: {avg_valid_loss:.4f}, Final Validation Accuracy: {valid_accuracy:.4f}"
            )

    def _tent(self, model: nn.Module):
        """
        Test-time Entropy Minimization (TENT) as proposed by Wang et al. (2020).
        Adapts only the affine parameters of BatchNorm layers, using entropy loss.
        """
        args = self.args
        logger.info("Starting TENT (Test-Time Adaptation)...")

        logger.info("Loading full dataset...")
        test_dataset = MMAC2023Task1Dataset(
            images_dir=args.validation_images_dir,
            labels_path=args.validation_labels_path,
            transform=[AugmentationTransform(), CenterCropTransform()],
            output_size=args.image_size,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 2 or 1,
        )

        # 1. Set model to train mode (needed for BN adaptation)
        model.train()

        # 2. Freeze all parameters except BN affine parameters
        for name, param in model.named_parameters():
            param.requires_grad = False  # default freeze

        bn_params = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = False  # disable running mean/var updates
                module.training = True  # ensure BN in training mode
                if module.affine:
                    if module.weight is not None:
                        module.weight.requires_grad = True
                        bn_params.append(module.weight)
                    if module.bias is not None:
                        module.bias.requires_grad = True
                        bn_params.append(module.bias)

        if not bn_params:
            logger.warning("No BatchNorm affine parameters found for adaptation.")
            return

        # 3. Optimizer only for BN affine parameters
        tent_optimizer = optim.SGD(bn_params, lr=args.tent_learning_rate)

        # 4. Entropy loss function
        def entropy_loss(outputs):
            probs = torch.softmax(outputs, dim=1)
            log_probs = torch.log(probs + 1e-6)
            return -torch.sum(probs * log_probs, dim=1).mean()

        # 5. Adaptation loop
        for epoch in range(args.tent_epochs):
            running_loss = 0.0
            total = 0
            for batch_idx, (inputs, _) in enumerate(test_dataloader):
                inputs = inputs.to(args.device)

                for _ in range(args.tent_steps):
                    tent_optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = entropy_loss(outputs)
                    loss.backward()
                    tent_optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

            avg_loss = running_loss / total
            logger.info(
                f"TENT Epoch {epoch+1}/{args.tent_epochs}. Avg Entropy Loss: {avg_loss:.4f}"
            )

        logger.info("TENT adaptation complete.")

    def greedy_soup(self):
        """
        Performs the Greedy Soup algorithm to combine multiple models.
        All model-specific configurations (e.g., model_name, num_classes) are
        taken from the Trainer's args.
        """
        args = self.args

        _, eval_loader, _ = self._get_dataloader()

        # Criterion is needed for evaluation
        criterion = self._criterion

        path_string = input("Please enter a comma-separated list of paths: ")
        model_paths = path_string.split(",")
        model_paths = [path.strip() for path in model_paths]

        if not model_paths:
            logger.warning("No model paths provided for Greedy Soup. Exiting.")
            return

        logger.info("Starting Greedy Soup algorithm...")

        # Initialize soup_model_state_dict to store the averaged weights
        soup_model_state_dict = None
        current_best_accuracy = (
            -1.0
        )  # Initialize with a value lower than any possible accuracy
        selected_models_count = 0

        # Store loaded candidate model state_dicts to avoid reloading
        candidate_state_dicts = []
        candidate_names = []  # To keep track of original paths for logging

        for path in model_paths:
            if not os.path.exists(path):
                logger.warning(f"Model checkpoint not found at {path}. Skipping.")
                continue

            # Create a fresh model instance and load its state_dict
            temp_model = self._get_model_no_logging()
            try:
                temp_model.load_state_dict(torch.load(path, map_location=args.device))
                candidate_state_dicts.append(temp_model.state_dict())
                candidate_names.append(path)
                logger.info(f"Loaded candidate model state from: {path}")
            except Exception as e:
                logger.error(
                    f"Failed to load state_dict from {path}: {e}. Skipping this model."
                )

        if not candidate_state_dicts:
            logger.warning("No valid candidate models to form a soup. Exiting.")
            return

        # Keep track of which candidate models have been selected by their original index
        selected_indices = []

        # Greedy selection loop
        # The loop runs up to the number of candidate models, or until no improvement.
        for _ in range(len(candidate_state_dicts)):
            best_ingredient_idx_this_iter = -1
            best_temp_accuracy_this_iter = -1.0

            # Iterate through remaining candidate models to find the best ingredient to add
            for j in range(len(candidate_state_dicts)):
                if j in selected_indices:
                    continue  # Skip already selected models

                temp_soup_model_state_dict = {}

                # Calculate the averaged state dict if candidate_state_dicts[j] were added
                num_models_in_proposed_soup = selected_models_count + 1

                for key in candidate_state_dicts[j].keys():
                    if (
                        soup_model_state_dict is None
                    ):  # This happens for the very first model selected
                        temp_soup_model_state_dict[key] = candidate_state_dicts[j][key]
                    else:
                        temp_soup_model_state_dict[key] = (
                            soup_model_state_dict[key] * selected_models_count
                            + candidate_state_dicts[j][key]
                        ) / num_models_in_proposed_soup

                # Create a temporary model and load the proposed soup state dict
                temp_model_for_eval = self._get_model_no_logging()
                temp_model_for_eval.load_state_dict(temp_soup_model_state_dict)
                temp_model_for_eval.to(args.device)

                # Evaluate this temporary soup model
                with temp_eval(temp_model_for_eval) as eval_model:
                    _, temp_accuracy = self._evaluation(
                        eval_loader, eval_model, criterion
                    )

                logger.info(
                    f"Proposing to add '{candidate_names[j]}'. Proposed Soup Accuracy: {temp_accuracy:.4f}"
                )

                # Check if this candidate improves the current best accuracy achievable in this iteration
                if temp_accuracy > best_temp_accuracy_this_iter:
                    best_temp_accuracy_this_iter = temp_accuracy
                    best_ingredient_idx_this_iter = j

            # If adding any model improves accuracy compared to the *overall* current best soup, commit
            if (
                best_ingredient_idx_this_iter != -1
                and best_temp_accuracy_this_iter > current_best_accuracy
            ):
                selected_indices.append(best_ingredient_idx_this_iter)
                selected_models_count += 1

                # Update soup_model_state_dict with the actual best ingredient found in this iteration
                if (
                    soup_model_state_dict is None
                ):  # Should only happen for the first selected model
                    soup_model_state_dict = candidate_state_dicts[
                        best_ingredient_idx_this_iter
                    ]
                else:
                    # Recompute the average for the current soup with the newly added model
                    for key in soup_model_state_dict.keys():
                        soup_model_state_dict[key] = (
                            soup_model_state_dict[key] * (selected_models_count - 1)
                            + candidate_state_dicts[best_ingredient_idx_this_iter][key]
                        ) / selected_models_count

                current_best_accuracy = best_temp_accuracy_this_iter
                logger.info(
                    f"GREEDY SOUP PROGRESS: Added '{candidate_names[best_ingredient_idx_this_iter]}'. New Best Soup Accuracy: {current_best_accuracy:.4f}"
                )
            else:
                logger.info(
                    f"No further improvement by adding remaining models or no valid models to add. Greedy Soup finished."
                )
                break  # No improvement, stop adding models

        if soup_model_state_dict is not None:
            # Save the final greedy soup model
            soup_model_save_name = f"greedy_soup_best_model_{args.model_name}.pth"
            soup_model_save_path = os.path.join(args.output_dir, soup_model_save_name)

            final_soup_model = (
                self._get_model_no_logging()
            )  # Initialize a fresh model for the soup
            final_soup_model.load_state_dict(soup_model_state_dict)
            torch.save(final_soup_model.state_dict(), soup_model_save_path)
            logger.info(f"Final Greedy Soup model saved to {soup_model_save_path}")
            logger.info(
                f"Final Greedy Soup Accuracy achieved: {current_best_accuracy:.4f}"
            )
        else:
            logger.warning(
                "Greedy Soup did not result in any combined model (perhaps no models improved accuracy)."
            )
