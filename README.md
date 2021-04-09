# MNIST_MultiLayer-Perceptron
A simple Python script to train a neural network with the MNIST dataset using the PyTorch framework.

### Prerequisites
Bellow There are steps for installing all the prerequites to run this Python program. I recommend creating a virtual environment to avoid version conflict. The steps are done in Linux using Pip. If you are not in Linux or using Pip, check requirements.txt for a list of needed package/modules.

Install the Pytorch with:

  ```sh
  pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  
This will download and install PyTorch and its prerequisites.
  
Install the tqdm loading bar:
  
  ```sh
  pip install tqdm
  ```
  
After this you can just run the script normally:
  
  ```sh
  python -m script.py
  ```
 
### What does it do?
The script will download the MNIST dataset and use it to train the parameters of a neural network, and then finally will print its precision.


### Neural Network Details
  This neural network is a relatively simple multilayer perceptron model.
  
*  The input has 784 values, that are equivalent to each grayscale pixel from each of the images.
*  This input gets through a linear transformation that results in the 500 values of the first hidden layer.
*  These 500 values are then sent through a ReLU (max(x,0)), resulting in 500 values of the second hidden layer.
*  Finally these 500 values are sent through another linear transformation, resulting in 10 different values. Each of these values represent a different number, and the higher the value, the higher the chance of the image being that specific number.
