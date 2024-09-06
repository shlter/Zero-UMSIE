# Zero-UMSIE


<div align=center><img src="img/1.png" height = "60%" width = "60%"/></div>

## Introduction
In this project, we use Ubuntu 16.04.5, Python 3.7, Pytorch 1.12.0 and one NVIDIA RTX 2080Ti GPU. 

## Datasets and results
Training dataset, testing dataset, and our predictions are available at [Google Drive]().

### Testing

The pretrained model is in the ./weights.

Check the model and image pathes in eval.py, and then run:

```
python new_test.py
```

### Training

To train the model, you need to prepare our training dataset.

Check the dataset path in main.py, and then run:
```
python train.py
```
