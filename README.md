# Data Scientist Nanodegree
# Deep Learning
## Project: Image Classifier

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project requires **Python 3**, **Jupyter Notebook** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [time](https://pypi.org/project/times/)
- [Pillow](https://pypi.org/project/Pillow/2.2.1/)
- [collections]: already included with in python
- [Pytorch](https://pytorch.org/)
- [torchvission](https://pypi.org/project/torchvision/)


## Project Motivation<a name="motivation"></a>

In this project, an image classifier is trained to recognize different species of flowers. Three models were tested to find the best model based on accuracy. During this process Linear, ReLU and LogSoftmax functions were applied to generate neural networks and trained using forward propagation and backpropagation.

## File Descriptions <a name="files"></a>

There are 4 files available here to showcase work related to the above questions. 
1. "Image Classifier Project.ipynb": This files contains the steps followed and code pursued. Markdown cells were used to assist in walking through the thought process for individual steps.
2. cat_to_name.json: This file contains the labels for different kind of flower species.
3. Image_Classification_checkpoint.pth: trained model
4. train.py: python script to build, train and save the model.
5. predict.py: python script to process image and make predictions.
6. README.md: This file gives an overview about the project, data used and the packages or libraries used in this project. 
7. Data files: 3 sample images are provided in "flowers" folder for testing purpose. 
8. Image Classifier Project.html: This file is just on export of notebook "Image Classifier Project.ipynb". 

## Results<a name="results"></a>

The Image Classifier predicted the correct name for the flower in the image passed as input.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Department of Engineering Science, University of Oxford  for the data. You can find the Licensing for the data and other descriptive information at the "Visual Geometry Group" link available [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 

