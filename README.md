# MICCAI 2023 MMAC Task1

## Getting started

Download dataset:

```bash
wget -O task1.zip "https://zenodo.org/records/11025749/files/1.%20Classification%20of%20Myopic%20Maculopathy.zip?download=1"
unzip "task1.zip"
```

Run:

```bash
python train.py
```

Submit:

```bash
cp model_outputs/resnet50_classifier_best.pth submit/resnet50_classifier_best.pth
zip -r submit.zip submit/*
```

## Result

Model 
ResNet18 10.7M 0.7823
ResNet50 22.4M 0.7540
MobileNetV3s 4.0M 0.7500
