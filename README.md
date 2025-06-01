# MICCAI 2023 MMAC Task1

## Getting started

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
zip -r submit.zip submit/*
```

If you are training a RETFound model and need to load from a pre-trained checkpoint, please obtain checkpoint access from Hugging Face first, then execute `huggingface-cli login --token <yourtoken>`.

## Result

Model 
ResNet18 10.7M 0.7823
ResNet50 22.4M 0.7540
MobileNetV3s 4.0M 0.7500
