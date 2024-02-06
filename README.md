# SAR Flood Segmentation using UNet 

This repository contains code for training and evaluating a UNet-based flood mapping segmentation model using a portion of the MMFlood dataset. The UNet architecture is a popular choice for semantic segmentation tasks due to its effectiveness in capturing spatial information.
![immagine](https://github.com/makovez/flood-mapping-sar/assets/21694707/9dea65ce-41f5-44b1-a1d8-e6e605df9f78)

## MMFlood Dataset
The MMFlood-cd dataset is a collection of satellite images labeled with flood extents. In this project, we utilize a subset of this dataset for training and evaluating our segmentation model.
The dataset can be seen [here](https://github.com/edornd/mmflood/)

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- MMFlood dataset (not provided, please obtain separately)

## Usage
1. **Data Preparation**: Obtain the MMFlood dataset and preprocess it as necessary. Ensure that the data is split into training and testing sets.

2. **Training**: Run the `train.py` script to train the UNet model on the training data. Adjust hyperparameters as needed. Example usage:

```
python train.py --data_path /path/to/training/data --epochs 50 --batch_size 8 --lr 0.001 --patch_size 128
```

3. **Evaluation**: After training, use the `evaluate.py` script to evaluate the trained model on unseen data. Example usage:

```
python evaluate.py --model_path /path/to/trained/model --data_path /path/to/test/data --patch_size 128 --save_images
```

## Directory Structure
- `data/`: Directory to store dataset.
- `models/`: Directory to store trained models.
- `train.py`: Script for training the UNet model.
- `evaluate.py`: Script for evaluating the trained model on unseen data.
