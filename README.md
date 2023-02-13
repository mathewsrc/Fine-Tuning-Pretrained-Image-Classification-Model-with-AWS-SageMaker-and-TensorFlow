# **Image-Classification-using-AWS-SageMaker**
Use AWS SageMaker to finetune a pretrained tensorflow model  that can perform image classification

### SageMaker Features used in this project:

- Sagemaker profiling
- Debugger
- Hyperparameter tuning

## Dataset

The PlantVillage dataset consists of 54303 images of healthy and unhealthy leaves, divided into 38 categories by species and disease. This dataset data is structure as follows:

``` 
FeaturesDict({
    'image': Image(shape=(None, None, 3), dtype=uint8),
    'image/filename': Text(shape=(), dtype=string),
    'label': ClassLabel(shape=(), dtype=int64, num_classes=38),
})
```

Where image field contains the Image itself, the image/filename contains the image filename and the label contains 38 different categories. 

![Example](https://github.com/punkmic/Image-Classification-using-AWS-SageMaker/blob/master/images/plants.PNG)

For more information see: [An open access repository of images on plant health to enable the development of mobile disease diagnostics](https://arxiv.org/abs/1511.08060).

For details about this dataset see: https://www.tensorflow.org/datasets/catalog/plant_village?hl=pt-br

Donwload dataset: [Plant_leaf_diseases_dataset_without_augmentation.zip](https://data.mendeley.com/datasets/tywbtsjrjv/1)

Convert jpg to jpeg
https://gist.github.com/ARolek/9199329

I needed a newer version of ImageMagick than is available on the yum packages on Amazon Linux. I tried using the remi repo but it failed with dependency errors. Here is what I did to install ImageMagick with support for PNG, JPG, and TIFF.

download the most recent package

wget http://www.imagemagick.org/download/ImageMagick.tar.gz
uncomress the package

tar -vxf ImageMagick.tar.gz
install the devel packages for png, jpg, tiff. these are dependencies of ImageMagick

sudo yum -y install libpng-devel libjpeg-devel libtiff-devel
configure ImageMagick without X11. this is a server without a display (headless) so we don't need X11

cd ImageMagick
./configure --without-x
make && make install
mission complete.

Tensorflow transfer-learning
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
