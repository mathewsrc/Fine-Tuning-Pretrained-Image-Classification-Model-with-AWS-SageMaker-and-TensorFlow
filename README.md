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

For more information see: [An open access repository of images on plant health to enable the development of mobile disease diagnostics](https://arxiv.org/abs/1511.08060).

For details about this dataset see: https://www.tensorflow.org/datasets/catalog/plant_village?hl=pt-br

Donwload dataset: [Plant_leaf_diseases_dataset_without_augmentation.zip](https://data.mendeley.com/datasets/tywbtsjrjv/1)

