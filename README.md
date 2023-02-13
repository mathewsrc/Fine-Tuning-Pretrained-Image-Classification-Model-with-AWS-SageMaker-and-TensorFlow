# **Fine-Tuning Pretrained Image Classification Model with AWS SageMaker and TensorFlow**


Project developed for AWS Machine Learning Engineer Scholarship offered by Udacity (2023)

In recent years, deep learning has revolutionized the field of computer vision with its ability to accurately classify images. One of the most popular techniques for image classification is using convolutional neural networks (CNNs), which have shown excellent results in comparison with others approaches such as full connected neural networks. However, training these models from scratch can be computationally intensive and time-consuming. To overcome this, another approach called transfer learning has been used and has become increasingly popular.

This project, uses Amazon Web Services (AWS) SageMaker and Tensorflow to fine-tune a pretrained model for binary image classification. The dataset used in this project can be found at https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification. In addition, SageMaker Debugger was used to measure performance of training job, system resource usage, and for framework metrics analysis.

<img src="https://user-images.githubusercontent.com/94936606/218581977-57d269fe-592f-4cc1-8812-1bc5d395d758.png" width=50% height=50%>
Source: DALL-E


## Project pipeline

<img src="https://user-images.githubusercontent.com/94936606/218553342-e0b3b855-6cc7-4ef2-a87f-0a24fe8b415c.jpg" width=90% height=90%>

## Project Features

- AWS SageMaker
- AWS SageMaker Debugger
- Tensorflow Framework version 2.9

## Dataset

The IS THAT SANTA? (Image Classification) dataset consists of 1230 images of Santa Claus and random images. This dataset is structured as follows:

![image](https://user-images.githubusercontent.com/94936606/218476207-78fa33e8-4da5-4470-9ef4-d3c26a402cf9.png)

For more information see: [IS THAT SANTA? (Image Classification)](https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification)

## Setup

### AWS SageMaker

```
Notebook enviroment (Kernel)

Image: Tensorflow 2.10.0 Python 3.9 CPU optimized
Instance type: ml.t3.medium
```

### Kaggle

Before we can access and download the Kaggle dataset, it is necessary to have a [Kaggle account](https://www.kaggle.com/) and a Kaggle API token (https://www.kaggle.com/account). Next paste the kaggle.json file in aws as follows and execute the code snipped below to move file to root:


<img src="https://user-images.githubusercontent.com/94936606/218479832-34f2ac1b-a7f8-4baa-b9f3-92e5a2e190cb.png" width=30% height=30%>


Move file to project root
```
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

The code bellow shows how to donwload dataset and unzip file

```
kaggle datasets download -d deepcontractor/is-that-santa-image-classification
```

```
unzip is-that-santa-image-classification.zip
```

As Tensorflow does not support jpg we need to convert images from jpg to jpeg. For this step, we can use a bash script that uses ImageMagick (https://imagemagick.org/index.php).

To install ImageMagick run the following code developed by ARolek (https://gist.github.com/ARolek/9199329) on terminal:

Download the most recent package

```
wget http://www.imagemagick.org/download/ImageMagick.tar.gz
```

Uncompress the package

```
tar -vxf ImageMagick.tar.gz
```

Install the devel packages for png, jpg, tiff. these are dependencies of ImageMagick

```
sudo yum -y install libpng-devel libjpeg-devel libtiff-devel
```

Configure ImageMagick without X11. this is a server without a display (headless) so we don't need X11

```
cd ImageMagick
./configure --without-x
make && make install
```

Now we can use a bash script on terminal to convert images:

```
./convert_jpg_to_jpeg.sh -d -r is_that_santa/
```

Note: the dataset name was manually renamed to is-that-santa. The -d flag in convert_jpg_jpeg.sh stands for delete the orinal images and the -r for recursively converts images.

Now we can upload files to AWS s3:
```
aws s3 cp is_that_santa s3://{bucke-name}/datasets/ --recursive > /dev/null
```

Note: replace the {bucket-name} with your own bucket name. --recursive > /dev/null is optinal.


### Python requirements and install

Requirements

```
tensorflow==2.10.1
smdebug==1.0.12
kaggle==1.5.12
```

Install

```
pip install -r requirements.txt
```

Or with MakeFile make command

```
make install
```

## Model prediction

<img src="https://user-images.githubusercontent.com/94936606/218544242-897c20d8-5575-4dc8-a74f-c0e258a221ec.PNG" width=50% height=50%>



Debugger and Profiler 
<img src="https://user-images.githubusercontent.com/94936606/218580803-80c429ab-3cb5-4a2a-8f59-e79f38e51891.png" width=50% height=50%>


<img src="https://user-images.githubusercontent.com/94936606/218581016-d6db88b1-30e0-418b-939d-ccdd6f79541f.png" width=50% height=50%>


<img src="https://user-images.githubusercontent.com/94936606/218581427-23d8a29b-6f4c-41e8-a211-f9c58b91d136.png" width=50% height=50%>


<img src="https://user-images.githubusercontent.com/94936606/218581708-d5db56a1-682c-4646-96e2-b7402ef6e75c.png" width=50% height=50%>



## References

https://www.tensorflow.org/tutorials/images/transfer_learning

https://www.tensorflow.org/tutorials/quickstart/advanced

https://docs.aws.amazon.com/sagemaker/latest/dg/model-access-training-data.html

https://github.com/aws/amazon-sagemaker-examples/blob/main/hyperparameter_tuning/

https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_gradienttape.py

https://github.com/awslabs/sagemaker-debugger/blob/master/docs/tensorflow.md

https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_debugger.rst

https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html

https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#tensorflow-specific-hook-api

https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-enable-tensorboard-summaries.html

https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-analyze-data.html

https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-access-data-profiling-default-plot.html

https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html
