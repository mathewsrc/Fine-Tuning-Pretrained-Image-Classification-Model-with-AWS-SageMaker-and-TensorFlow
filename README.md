# **Fine-Tuning Pretrained Image Classification Model with AWS SageMaker and TensorFlow**

In recent years, deep learning has revolutionized the field of computer vision with its ability to accurately classify images. One of the most popular techniques for image classification is using convolutional neural networks (CNNs), which have shown excellent results in comparison with others approaches such as full connected neural networks. However, training these models from scratch can be computationally intensive and time-consuming. To overcome this, another approach called transfer learning has been used and has become increasingly popular.

This project, uses Amazon Web Services (AWS) SageMaker and Tensorflow to fine-tune a pretrained model for binary image classification. The dataset used in this project can be found at https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification. In addition, SageMaker Debugger was used to measure performance of training job, system resource usage, and for framework metrics analysis.


### Project Features:

- AWS SageMaker
- AWS SageMaker Debugger
- Tensorflow version 2.10

## Dataset

The IS THAT SANTA? (Image Classification) dataset consists of 1230 images of Santa Claus and random images. This dataset is structured as follows:

![image](https://user-images.githubusercontent.com/94936606/218476207-78fa33e8-4da5-4470-9ef4-d3c26a402cf9.png)

For more information see: [IS THAT SANTA? (Image Classification)](https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification)

### Setup

AWS SageMaker

```
Notebook enviroment (Kernel)

Image: Tensorflow 2.10.0 Python 3.9 CPU optimized
Instance type: ml.t3.medium
```

Kaggle

Before we can access and download the Kaggle dataset, it is necessary to have a [Kaggle account](https://www.kaggle.com/) and a Kaggle API token (https://www.kaggle.com/account). Next paste the kaggle.json file in aws as follows and execute the code snipped below to move file to root:

![image](https://user-images.githubusercontent.com/94936606/218479832-34f2ac1b-a7f8-4baa-b9f3-92e5a2e190cb.png)

```
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

The code bellow shows how to donwload dataset and unzip file.

```
!kaggle datasets download -d deepcontractor/is-that-santa-image-classification
```

```
!unzip is-that-santa-image-classification.zip
```

As Tensorflow does not supports jpg we need to convert images from jpg to jpeg. For this step we going to use a bash script that uses ImageMagick (https://imagemagick.org/index.php).

To install ImageMagick run the following code developed by ARolek (https://gist.github.com/ARolek/9199329) on terminal:

Download the most recent package

```
wget http://www.imagemagick.org/download/ImageMagick.tar.gz
```

Uncomress the package

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

Now we can use this bash script on terminal to convert images:

```
!./convert_jpg_to_jpeg.sh -d -r is_that_santa/
```

Note: the dataset name was manually renamed to is-that-santa. The -d flag in convert_jpg_jpeg.sh stands for delete the orinal images and the -r for recursively converts images.


Python requirements and install

Requirements

tensorflow
smdebug
kaggle

Install

```
!pip install -r requirements.txt
```







