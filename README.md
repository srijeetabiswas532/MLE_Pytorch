# MLE_Pytorch
Creating several Pytorch functionalities from scratch.
A Pytorch framework from scratch, encompassing tensor operations, auto-differentiation, and advanced
features for deep learning model development. Enhanced ML capabilities by implementing model debugging, testing, visualization, GPU utilization, compression techniques, and efficient low-power inference methods.
Individual READMEs for each module in the respective folders.

This repository is my personal implementation of MiniTorch, as detailed in the official MiniTorch [documentation](https://minitorch.github.io/).

Setup
Follow the installation guide [here](https://minitorch.github.io/install/).

## Training
Visualize training using Streamlit with this command:

streamlit run app.py -- [module number]
This project also implemented a version of LeNet on MNIST: a classic convolutional neural network (CNN) for digit recognition, and for a 1D conv for NLP sentiment classification.

You can run NLP and CV training scripts directly from the command line:

For NLP training:

python project/run_sentiment.py
For CV training:

python project/run_mnist_multiclass.py
Assignment
Please refer to the README.md in each module.
