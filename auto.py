import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split

from dataset import MMAC2023Task1Dataset, AugmentationTransform, CenterCropTransform, RandomCropTransform

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

train_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
    images_dir=training_set_images_dir,
    labels_path=training_set_labels_path,
    transform=[],
).get_image_label_pair_df()

val_dataset: MMAC2023Task1Dataset = MMAC2023Task1Dataset(
    images_dir=validation_set_images_dir,
    labels_path=validation_set_labels_path,
    transform=[],
).get_image_label_pair_df()

# Initialize the ImagePredictor
predictor = MultiModalPredictor(label="label", path="output/auto")

print(train_dataset)

print("\nStarting AutoGluon training...")
predictor.fit(
    train_dataset,
    tuning_data=val_dataset, # Use the validation data for early stopping and hyperparameter tuning
    time_limit=360 # 1 hour
)

# Evaluate the model on the validation set
print("\nEvaluating the model on the validation set...")
val_accuracy = predictor.evaluate(val_dataset)
print(f"Validation Accuracy: {val_accuracy}")

# Make predictions on new data (example with validation data)
print("\nMaking predictions on the validation set...")
predictions = predictor.predict(val_dataset)
print("Example predictions:")
print(predictions.head())

# To get prediction probabilities
probabilities = predictor.predict_proba(val_dataset)
print("\nExample prediction probabilities:")
print(probabilities.head())

# To save the predictor for later use
predictor.save(os.path.join("output/auto", 'my_image_predictor'))

# To load a saved predictor
# loaded_predictor = ImagePredictor.load(os.path.join(output_dir, 'my_image_predictor'))

print("\nAutoGluon training complete!")