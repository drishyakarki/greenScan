# GreenScan - Plant Disease Detection System

In this plant disease detection I, have covered the following topics
## Introduction
In this project I have achieved the following objectives:
- Understanding Pytorch Tensor
- Created custom Datasets and DataLoaders in Pytorch
- Used Transforms of Pytorch and visualisation
- Created a model and build it (used ResNet as base model)
- Understanding the automatic differentiation
- Optimizing model parameters
- Saving and loading the model

## Project Structure
```
.
├── data
│   └── New Plant Diseases Dataset(Augmented)
├── inference.py
├── models
│   ├── custom_resnet_model2.pth
│   └── custom_resnet_model.pth
├── notebook
│   └── notebook.ipynb
├── plots
│   └── plotLossAcc.png
├── README.md
├── requirements.txt
├── src
│   ├── customDataset.py
│   ├── model.py
│   ├── __pycache__
│   └── train.py
├── test
│   └── test
```

*You can download the dataset used (i.e.PlantVillage Dataset) from kaggle or any other sites*

## Understanding the Pytorch Tensor
Pytorch Tensors are used to represent numerical data efficiently. They can handle various data types. Tensors are the building blocks of deep learning models. Pytorch seamlessly integrates with GPUs, enabling computations on these devices using tensors. Pytorch tensors are easily converted to and from NumPy arrays, facilitating interoperability with other computing linraries.

## Loading Datasets and DataLoaders 
PyTorch provides built-in utilities for loading and managing datasets. You can use predefined datasets like CIFAR-10, ImageNet, or custom datasets. Additionally, you can create custom dataset classes by inheriting from torch.utils.data.Dataset. These classes define how to load and preprocess your data.

Once you have a dataset, you can create a DataLoader. The DataLoader is responsible for batching, shuffling, and efficiently loading data in parallel during training and evaluation. It's a crucial component for handling large datasets and ensuring that data is fed to your model in an optimized way.DataLoaders divide your dataset into mini-batches, making it feasible to train deep learning models on limited computational resources like GPUs.

*I have created the custom dataset to extract the images and labels from the dataset. It is inside the src folder.*

## Using Transforms of PyTorch and Data Visualization
Data preprocessing is typically performed using **transforms**. Transforms allow you to apply various operations to your data, such as resizing images, normalizing pixel values, or augmenting data for better model training. PyTorch's torchvision.transforms module provides a range of transformations to choose from. Transforms ensure that data is consistently preprocessed, making it easier to work with various datasets and models.

Data visualization helps you understand your model's performance and behavior. You can create plots and charts to visualize metrics like loss curves, accuracy, and confusion matrices, which provide insights into model training.

*Inside the notebook you can see the different visualizations and data analysis I have performed before proceeding to model creation*

## Creating a model and Building
In your project, creating and building deep learning models using PyTorch is a pivotal step in achieving accurate and reliable results. This process encompasses defining the model architecture, selecting appropriate neural network layers, initializing model parameters, and deciding whether to employ transfer learning techniques. The model architecture must align with the specific task, be it image classification, object detection, or natural language processing. PyTorch's flexibility allows you to create custom layers and architectures tailored to your project's unique requirements. Additionally, you configure the model to run on either a GPU or CPU, optimizing its performance through parallel processing on GPUs.

Once the model architecture is established, you assemble the model by composing the layers sequentially or designing complex architectures. Key considerations include choosing an appropriate loss function, selecting an optimizer for parameter updates, and setting hyperparameters like learning rate and batch size. The training loop is implemented to feed data batches, compute gradients, and iteratively update model parameters, ultimately optimizing the model's ability to make accurate predictions. Post-training evaluation on validation and test datasets ensures that the model generalizes effectively to unseen data, demonstrating the critical role that creating and building models play in your project's success.

## Understanding the automatic differentiation
Automatic differentiation, often referred to as autograd (short for automatic gradient), is a fundamental concept in deep learning frameworks like PyTorch. It is a critical component that enables the training of neural networks through backpropagation.

Automatic differentiation is the process of computing gradients or derivatives of functions with respect to their inputs. In the context of deep learning, it involves computing gradients of the loss function with respect to the model's parameters. These gradients indicate how the loss changes concerning each parameter and guide the optimization process.

## Optimizing model parameters
Optimization involves the process of finding the best set of model parameters that minimize a predefined loss function. The loss function quantifies how well the model's predictions match the actual target values. The goal is to adjust the model's weights and biases in such a way that the loss is minimized. This is typically done through an iterative process called gradient descent. PyTorch provides various optimization algorithms, such as stochastic gradient descent (SGD), Adam, and RMSprop, which automatically update model parameters based on the gradients computed during backpropagation. These algorithms help the model converge towards the optimal parameter values, resulting in improved accuracy and generalization.

Hyperparameter tuning is another critical aspect of model optimization. Hyperparameters, like learning rate, batch size, and the choice of optimizer, are parameters that are not learned from the data but need to be set manually. Finding the right combination of hyperparameters can significantly impact the training process and the final model's performance. Techniques like grid search or random search are often used to explore different hyperparameter configurations and identify the ones that yield the best results. Overall, optimizing model parameters and hyperparameters is a fundamental part of building effective deep learning models in your project.

## Saving and Loading the model
After training your plant disease detection model, you can save it to disk using PyTorch's built-in serialization methods. This process involves saving the model architecture, learned weights, and other necessary information. Saved models can be used for various purposes, such as making predictions on new data, fine-tuning, or sharing with collaborators. This step ensures that your model's valuable knowledge is preserved and can be easily accessed in the future.

When you need to use a pre-trained model, either for inference or further training, you can load it back into your project using PyTorch's model loading functionality. This process restores the model's architecture and parameters, making it ready for use. Loading pre-trained models is especially useful if you want to build on existing research or leverage transfer learning, where you start with a pre-trained model and fine-tune it for your specific task. It saves time and computational resources, as you don't have to retrain the model from scratch.

*In this plant disease detection system, I have covered all the above mentioned topics and implemented them into real application. You can checkout the accuracy and loss plot inside the plots directory*

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: Your system should have Python installed. You can download and install Python from [python.org](https://www.python.org/downloads/).

- **Virtual Environment** (Optional but recommended): It's a good practice to create and activate a virtual environment for this project to isolate dependencies. You can create a virtual environment using Python's built-in `venv` module. If you're using Python 3.10 or later, you can run:

```
python -m venv env
```
Activate the virtual environment:

On Windows:
```
.\env\Scripts\activate
```
On macOS and Linux:

```
source env/bin/activate
```

**CMake:** This project requires CMake version 3.27.5 or higher. You can download CMake from cmake.org.

**Other Python Dependencies:** Install the required Python packages and libraries by running the following command in your virtual environment:

```
pip install -r requirements.txt
```
This will install all the necessary Python packages listed in the requirements.txt file.

*Once you have followed these steps, you should be able to run the project smoothly*

# Prepared By
Drishya Karki
