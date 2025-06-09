# --- AutoGluon Implementation ---

import os

import pandas as pd

from autogluon.vision import ImagePredictor
from sklearn.model_selection import train_test_split

from dataset import MMAC2023Task1Dataset, AugmentationTransform, CenterCropTransform, RandomCropTransform

# Define your data paths
# Make sure these paths are correct for your environment
images_dir = 'path/to/your/images'  # e.g., 'data/train_images'
labels_path = 'path/to/your/labels.csv'  # e.g., 'data/train_labels.csv'
output_dir = 'autogluon_output' # Directory to save AutoGluon models and results

# Initialize the dataset to get access to labels_df and image_files
# We don't need any transforms for this step, as AutoGluon handles its own transformations.
# But we need a list of callables for the transform parameter.
dummy_transform = []
full_dataset_info = MMAC2023Task1Dataset(images_dir=images_dir, labels_path=labels_path, transform=dummy_transform)

# Create a DataFrame for AutoGluon
# The 'image' column will contain full paths to the images
# The 'label' column will contain the corresponding grades
autogluon_df = pd.DataFrame({
    'image': [os.path.join(images_dir, img_name) for img_name in full_dataset_info.image_files],
    'label': [full_dataset_info.image_to_label[img_name] for img_name in full_dataset_info.image_files]
})

# Display the first few rows of the DataFrame to verify
print("DataFrame for AutoGluon:")
print(autogluon_df.head())
print(f"Total samples: {len(autogluon_df)}")

# Split data into training and validation sets
# AutoGluon can also perform internal validation splits, but explicitly splitting
# here gives you more control and visibility.
train_data, val_data = train_test_split(autogluon_df, test_size=0.2, random_state=42, stratify=autogluon_df['label'])

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Initialize the ImagePredictor
predictor = ImagePredictor(path=output_dir)

# Fit the predictor
# This step will train the model. AutoGluon will automatically
# select a model, optimize hyperparameters, and perform data augmentation.
# You can specify `hyperparameters` for different models or
# `time_limit` to control training duration.
print("\nStarting AutoGluon training...")
predictor.fit(
    train_data,
    tuning_data=val_data, # Use the validation data for early stopping and hyperparameter tuning
    hyperparameters={'epochs': 5, 'batch_size': 32}, # Example: run for 5 epochs
    # You can specify a list of models: hyperparameters={'model': ['resnet50', 'mobilenet_v3_large']}
    # Or set a time limit: time_limit=3600 # 1 hour
)

# Evaluate the model on the validation set
print("\nEvaluating the model on the validation set...")
val_accuracy = predictor.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy}")

# Make predictions on new data (example with validation data)
print("\nMaking predictions on the validation set...")
predictions = predictor.predict(val_data)
print("Example predictions:")
print(predictions.head())

# To get prediction probabilities
probabilities = predictor.predict_proba(val_data)
print("\nExample prediction probabilities:")
print(probabilities.head())

# To save the predictor for later use
predictor.save(os.path.join(output_dir, 'my_image_predictor'))

# To load a saved predictor
# loaded_predictor = ImagePredictor.load(os.path.join(output_dir, 'my_image_predictor'))

print("\nAutoGluon training complete!")