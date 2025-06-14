# MICCAI 2023 MMAC Task1

## Getting started

```bash
git clone https://github.com/xrysamuel/MMAC2023-Solution
cd MMAC2023-Solution
```

Download the dataset and extract it to the corresponding directory.

```bash
wget -O task1.zip "https://zenodo.org/records/11025749/files/1.%20Classification%20of%20Myopic%20Maculopathy.zip?download=1"
unzip "task1.zip"
```

Start training (using ResNet18 as an example; for other models, replace with the corresponding recipe configuration file).

```bash
python train.py recipes/resnet18.yaml
```

Submit by modifying the model class in model.py and copying the best model checkpoint to the submit folder.

```bash
MODEL_NAME=resnet18
zip -r submit/submit-${MODEL_NAME}.zip submit/submit-${MODEL_NAME}/*
```

If you are training a RETFound model and need to load from a pre-trained checkpoint, please obtain checkpoint access from Hugging Face first, then execute `huggingface-cli login --token <yourtoken>`.

Using notebook:

```python
import os
import sys
import logging
import argparse

from transformers import HfArgumentParser

from train import Trainer, TrainArgs

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # Remove all handlers associated with the root logger object.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = HfArgumentParser((TrainArgs))

try:
    train_args, = parser.parse_yaml_file(yaml_file="recipes/retfound.yaml")
except Exception as e:
    logging.error(f"Error parsing YAML file: {e}")
    exit()

logging.info(train_args)

trainer = Trainer(train_args)
trainer.train()
```

## 

## Best Result

https://codalab.lisn.upsaclay.fr/competitions/12441

ResNet18 -> 0.8564229321

```log
2025-06-02 01:35:57,253 - INFO - TrainArgs(images_dir='1. Classification of Myopic Maculopathy/1. Images/1. Training Set', labels_path='1. Classification of Myopic Maculopathy/2. Groundtruths/1. MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv', validation_images_dir='1. Classification of Myopic Maculopathy/1. Images/2. Validation Set', validation_labels_path='1. Classification of Myopic Maculopathy/2. Groundtruths/2. MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv', image_size=[224, 224], batch_size=32, num_epochs=50, learning_rate=0.0001, test_split_ratio=0.2, eval_steps=100, patience=3, num_classes=5, device='cuda', output_dir='output/resnet18', model_name='resnet18', model_save_name='resnet18_best.pth', random_seed=42, pretrained=True, train_test_split=False)
2025-06-02 01:35:57,255 - INFO - Set random seed to 42 for reproducibility.
2025-06-02 01:35:57,255 - INFO - Using device: cuda
2025-06-02 01:35:57,255 - INFO - Output directory 'output/resnet18' ensured.
2025-06-02 01:35:57,255 - INFO - Loading full dataset...
2025-06-02 01:35:57,264 - INFO - Loading validation dataset...
2025-06-02 01:35:57,267 - INFO - Training samples: 1143
2025-06-02 01:35:57,267 - INFO - Validation samples: 248
2025-06-02 01:35:57,267 - INFO - Initializing model...
2025-06-02 01:35:57,453 - INFO - Initialized ResNet18 with ImageNet pre-trained weights.
2025-06-02 01:35:57,453 - INFO - Modified final layer to have 5 output features for ResNet18.
2025-06-02 01:35:57,630 - INFO - {'Total Parameters (M)': 11.18, 'Trainable Parameters (M)': 11.18, 'Total Parameters (raw)': 11179077, 'Trainable Parameters (raw)': 11179077, 'Percentage Trainable': '100.0%'}
2025-06-02 01:35:57,630 - INFO - Model moved to cuda.
2025-06-02 01:35:57,630 - INFO - Starting training...
2025-06-02 01:36:04,133 - INFO - Epoch 1/50 finished. Train Loss: 0.7722, Train Accuracy: 0.6929
2025-06-02 01:36:09,966 - INFO - Epoch 2/50 finished. Train Loss: 0.4392, Train Accuracy: 0.8320
2025-06-02 01:36:16,523 - INFO - Step 100. Internal Test Loss: 0.8700, Internal Test Accuracy: 0.6815
2025-06-02 01:36:16,590 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.6815
2025-06-02 01:36:17,521 - INFO - Epoch 3/50 finished. Train Loss: 0.3565, Train Accuracy: 0.8644
2025-06-02 01:36:23,624 - INFO - Epoch 4/50 finished. Train Loss: 0.2680, Train Accuracy: 0.9081
2025-06-02 01:36:29,748 - INFO - Epoch 5/50 finished. Train Loss: 0.2345, Train Accuracy: 0.9108
2025-06-02 01:36:35,793 - INFO - Step 200. Internal Test Loss: 0.8838, Internal Test Accuracy: 0.7056
2025-06-02 01:36:35,868 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.7056
2025-06-02 01:36:37,793 - INFO - Epoch 6/50 finished. Train Loss: 0.1827, Train Accuracy: 0.9283
2025-06-02 01:36:43,767 - INFO - Epoch 7/50 finished. Train Loss: 0.1679, Train Accuracy: 0.9370
2025-06-02 01:36:49,254 - INFO - Epoch 8/50 finished. Train Loss: 0.1859, Train Accuracy: 0.9423
2025-06-02 01:36:53,599 - INFO - Step 300. Internal Test Loss: 0.7084, Internal Test Accuracy: 0.7460
2025-06-02 01:36:53,658 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.7460
2025-06-02 01:36:56,438 - INFO - Epoch 9/50 finished. Train Loss: 0.1392, Train Accuracy: 0.9571
2025-06-02 01:37:02,247 - INFO - Epoch 10/50 finished. Train Loss: 0.1336, Train Accuracy: 0.9493
2025-06-02 01:37:07,807 - INFO - Epoch 11/50 finished. Train Loss: 0.1043, Train Accuracy: 0.9606
2025-06-02 01:37:11,178 - INFO - Step 400. Internal Test Loss: 1.0292, Internal Test Accuracy: 0.7097
2025-06-02 01:37:11,179 - INFO - No improvement for 1 evaluations.
2025-06-02 01:37:15,017 - INFO - Epoch 12/50 finished. Train Loss: 0.0827, Train Accuracy: 0.9746
2025-06-02 01:37:20,605 - INFO - Epoch 13/50 finished. Train Loss: 0.0910, Train Accuracy: 0.9685
2025-06-02 01:37:26,773 - INFO - Step 500. Internal Test Loss: 0.8321, Internal Test Accuracy: 0.7581
2025-06-02 01:37:26,833 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.7581
2025-06-02 01:37:27,288 - INFO - Epoch 14/50 finished. Train Loss: 0.1108, Train Accuracy: 0.9606
2025-06-02 01:37:33,082 - INFO - Epoch 15/50 finished. Train Loss: 0.0967, Train Accuracy: 0.9659
2025-06-02 01:37:38,857 - INFO - Epoch 16/50 finished. Train Loss: 0.0839, Train Accuracy: 0.9694
2025-06-02 01:37:44,228 - INFO - Step 600. Internal Test Loss: 1.0678, Internal Test Accuracy: 0.7540
2025-06-02 01:37:44,228 - INFO - No improvement for 1 evaluations.
2025-06-02 01:37:45,673 - INFO - Epoch 17/50 finished. Train Loss: 0.0770, Train Accuracy: 0.9764
2025-06-02 01:37:51,327 - INFO - Epoch 18/50 finished. Train Loss: 0.0753, Train Accuracy: 0.9720
2025-06-02 01:37:56,920 - INFO - Epoch 19/50 finished. Train Loss: 0.0587, Train Accuracy: 0.9799
2025-06-02 01:38:01,577 - INFO - Step 700. Internal Test Loss: 1.1469, Internal Test Accuracy: 0.7460
2025-06-02 01:38:01,577 - INFO - No improvement for 2 evaluations.
2025-06-02 01:38:03,872 - INFO - Epoch 20/50 finished. Train Loss: 0.0881, Train Accuracy: 0.9668
2025-06-02 01:38:09,579 - INFO - Epoch 21/50 finished. Train Loss: 0.0769, Train Accuracy: 0.9720
2025-06-02 01:38:15,261 - INFO - Epoch 22/50 finished. Train Loss: 0.0464, Train Accuracy: 0.9843
2025-06-02 01:38:19,395 - INFO - Step 800. Internal Test Loss: 0.9846, Internal Test Accuracy: 0.7661
2025-06-02 01:38:19,474 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.7661
2025-06-02 01:38:22,792 - INFO - Epoch 23/50 finished. Train Loss: 0.0519, Train Accuracy: 0.9799
2025-06-02 01:38:28,858 - INFO - Epoch 24/50 finished. Train Loss: 0.0515, Train Accuracy: 0.9834
2025-06-02 01:38:36,339 - INFO - Step 900. Internal Test Loss: 0.8860, Internal Test Accuracy: 0.8024
2025-06-02 01:38:36,400 - INFO - Saved best model to output/resnet18/resnet18_best.pth with accuracy: 0.8024
2025-06-02 01:38:36,430 - INFO - Epoch 25/50 finished. Train Loss: 0.0432, Train Accuracy: 0.9869
2025-06-02 01:38:42,558 - INFO - Epoch 26/50 finished. Train Loss: 0.0634, Train Accuracy: 0.9808
2025-06-02 01:38:48,456 - INFO - Epoch 27/50 finished. Train Loss: 0.0526, Train Accuracy: 0.9799
2025-06-02 01:38:54,785 - INFO - Step 1000. Internal Test Loss: 0.9173, Internal Test Accuracy: 0.7903
2025-06-02 01:38:54,786 - INFO - No improvement for 1 evaluations.
2025-06-02 01:38:55,820 - INFO - Epoch 28/50 finished. Train Loss: 0.0378, Train Accuracy: 0.9860
2025-06-02 01:39:02,576 - INFO - Epoch 29/50 finished. Train Loss: 0.0343, Train Accuracy: 0.9886
2025-06-02 01:39:09,038 - INFO - Epoch 30/50 finished. Train Loss: 0.0518, Train Accuracy: 0.9834
2025-06-02 01:39:14,686 - INFO - Step 1100. Internal Test Loss: 0.9619, Internal Test Accuracy: 0.7782
2025-06-02 01:39:14,687 - INFO - No improvement for 2 evaluations.
2025-06-02 01:39:16,521 - INFO - Epoch 31/50 finished. Train Loss: 0.0623, Train Accuracy: 0.9790
2025-06-02 01:39:22,891 - INFO - Epoch 32/50 finished. Train Loss: 0.0492, Train Accuracy: 0.9816
2025-06-02 01:39:29,128 - INFO - Epoch 33/50 finished. Train Loss: 0.0554, Train Accuracy: 0.9843
2025-06-02 01:39:34,169 - INFO - Step 1200. Internal Test Loss: 1.1301, Internal Test Accuracy: 0.7258
2025-06-02 01:39:34,169 - INFO - No improvement for 3 evaluations.
2025-06-02 01:39:34,169 - INFO - Early stopping triggered!
2025-06-02 01:39:34,506 - INFO - Epoch 34/50 finished. Train Loss: 0.0367, Train Accuracy: 0.9870
2025-06-02 01:39:34,506 - INFO - Training complete.
2025-06-02 01:39:34,506 - INFO - Performing final evaluation on validation set.
/home/samuel/Projects/MMAC/train.py:235: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(best_model_path))
2025-06-02 01:39:34,560 - INFO - Loaded best model from output/resnet18/resnet18_best.pth for final validation.
2025-06-02 01:39:35,715 - INFO - Final Validation Loss: 0.8860, Final Validation Accuracy: 0.8024
```
