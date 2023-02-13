# **Fine-Tuning Pretrained Image Classification Model with AWS SageMaker and TensorFlow**

In recent years, deep learning has revolutionized the field of computer vision with its ability to accurately classify images. One of the most popular techniques for image classification is using convolutional neural networks (CNNs), which have shown excellent results in comparison with others approaches such as full connected neural networks. However, training these models from scratch can be computationally intensive and time-consuming. To overcome this, another approach called transfer learning has been used and has become increasingly popular.

This project, uses Amazon Web Services (AWS) SageMaker and Tensorflow to fine-tune a pretrained model for binary image classification. The dataset used in this project can be found at https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification. In addition, SageMaker Debugger was used to measure performance of training job, system resource usage, and for framework metrics analysis.


### Project Features:

- AWS SageMaker
- AWS SageMaker Debugger
- Tensorflow version 2.9

## Dataset

The IS THAT SANTA? (Image Classification) dataset consists of 1230 images of Santa Claus and random images. This dataset is structured as follows:

![image](https://user-images.githubusercontent.com/94936606/218476207-78fa33e8-4da5-4470-9ef4-d3c26a402cf9.png)

For more information see: [IS THAT SANTA? (Image Classification)](https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification)

