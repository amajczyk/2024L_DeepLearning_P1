
# Deep Learning - CNNs
## Project 1

Convolutional Neural Networks project for Deep Learning at Master's Data Science at Warsaw University of Technology. The project consists of considerations about different form of augmentations and accuracy of pretrained models such as Resnet50 and EfficientNetV2.
## Authors

- [Adam Majczyk](https://github.com/amajczyk)
- [Szymon Matuszewski](https://github.com/szymonsm)


## Prerequisites

Install needed dependencies:

```bash
  pip install -r requirements.txt
```
Now, you are able to regenerate the training pipeline.
## Documentation

Files in main directory:

- **BEST_ClassicCNN_keras_aug.ipynb**: Jupyter Notebook with the pipeline for training the best Classic CNN with augmentations

- **BEST_ClassicCNN_keras_hyperparams.ipynb**: Jupyter Notebook with the pipeline for checking the best hyperparameters (learning rate and optimizer) for the best Classic CNN 

- **ClassicCNN_keras_no_aug.ipynb**: Jupyter Notebook with the pipeline for training different architectures of Classic CNNs without augmentations

- **efficientLarge_keras_aug.ipynb**: Jupyter Notebook with the pipeline for training EfficientNetV2 model with augmentations (Flip, Gaussian Noise and Flip-Gaussian Noise combination) - 20 epochs training per iteration (*not mentioned in the report*)

- **efficientLarge_keras_aug_13epochs.ipynb**: Jupyter Notebook with the pipeline for training EfficientNetV2 model with augmentations (Flip, Gaussian Noise and Flip-Gaussian Noise combination) - 13 epochs training per iteration

- **efficientLarge_keras_no_aug.ipynb**: Jupyter Notebook with the pipeline for training EfficientNetV2 model without augmentations

- **ensemble_predictor.ipynb**: Jupyter Notebook with the ensamble predictor composed of the committee of several CNNs (NOTE: there, already trained models are referenced - it requires, that they are avaiable in `.h5` files - not uploaded as they are too large)

- **resnet_keras_aug.ipynb**: Jupyter Notebook with the pipeline for training Resnet50 model with augmentations (Flip, Gaussian Noise and Flip-Gaussian Noise combination) - 50 epochs training per iteration

- **resnet_keras_no_aug.ipynb**: Jupyter Notebook with the pipeline for training Resnet50 model without augmentations - 40 epochs training per iteration

Folders in main directory:

- **.LEGACY**: Folder containing preliminary tests on Classic CNNs done in PyTorch

- **PDFs**: Plan for a lab milestone 

- **results**: Folder with different accuracies and confusion matrices stored in pickle files

- **src**: Folder with implementation of Classic CNNs and augmentations used in Jupyter Notebooks

- **summaries**: Folder with different summaries per experiment
