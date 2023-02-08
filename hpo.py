#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torchvision
#import torchvision.models as models
#import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow import keras
from tensorflow import Adam
import tensorflow_datasets as tfds
import os
import boto3

import argparse

IMG_SIZE = 224 
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, base_model, train_ds, test_ds, criterion, optimizer, epochs, batch_size):
    # Compile the model with categorical crossentropy loss and the Adam optimizer
    model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])

    # Train the model on your own dataset
    history = model.fit(train_ds, test_ds, batch_size=batch_size, epochs=epochs)

    
def net():
    # Load the MobileNetV2 model with pre-trained weights
    base_model  = keras.applications.MobileNetV2(input_shape=(224,224,3), 
                                                 weights='imagenet',
                                                 include_top=False)
    # Freeze the base model layers to prevent updating their weights during training
    base_model.trainable = False    
    
    # Create a new model on top of the frozen base model
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def main(args):
    
    train_kwargs = {"batch_size": args.batch_size, 
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs
                    }
    
    test_kwargs = {"batch_size": args.batch_size,
                   "epochs": args.epochs}

    model=net()
    
    loss_criterion = 'sparse_categorical_crossentropy'
    
    optimizer = Adam()
    
    path_to_dataset = './plant_leaf_diseases_dataset_without_augmentation'
    
    builder = tfds.builder_from_directory(path_to_dataset)
    
    train_ds, test_ds = builder.as_dataset(split='test+train[:75%]',
                            as_supervised=True,
                            shuffle_files=True)
    
    model=train(model, train_ds, test_ds, loss_criterion, optimizer)
    
    #test(model, test_ds, loss_criterion)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    
    model.save('saved_model/my_model')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    args=parser.parse_args()
    
    main(args)
