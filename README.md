# Conditional WGAN-GP for Graph Generation Based on Descriptions

## Introduction

Implementation of a conditional Wiertrass GAN with Gradient Penalty for Graph generation given a description a graph. 

This work was developed as part of a private Kaggle competition hosted for the ALTEGRAD course by the MVA. The dataset is not publicly available.

Please refer to the report for further details on our experiments and results.

## Install

```
git clone https://github.com/MaxLSB/cWGAN-GP.git
```
```
pip install -r requirements.txt
```

## Training 

To train the model using the Kaggle Data, use the ```main.py``` file with argparse commands.

Example of a training command:
```
python main.py --batch_size 128 --noise_dim 64 --hidden_dim_generator 128 --data_aug 16000
```
_(The data folder should be located in the same directory as all the code files.)_

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Training Hyperparameters** | | | |
| `--batch_size` | int | 256 | Batch size for training. |
| `--generator_lr` | float | 0.0005 | Learning rate for the generator. |
| `--discriminator_lr` | float | 0.001 | Learning rate for the discriminator. |
| `--num_epochs` | int | 100 | Number of epochs to train. |
| **Model Architecture Hyperparameters** | | | |
| `--n_max_nodes` | int | 50 | Maximum number of nodes in the graph. |
| `--noise_dim` | int | 32 | Dimension of the noise vector. |
| `--cond_dim` | int | 7 | Conditioning dimension (7 graph features extracted from the prompt). |
| `--hidden_dim_generator` | int | 256 | Hidden dimension for the generator. |
| `--hidden_dim2_generator` | int | 128 | Second hidden dimension for the generator. |
| `--hidden_dim_discriminator` | int | 256 | Hidden dimension for the discriminator. |
| **Data Augmentation** | | | |
| `--data_aug` | int | 8000 | Number of graphs generated through data augmentation. |

_(Need to add the data folder directory in the argparse)_
