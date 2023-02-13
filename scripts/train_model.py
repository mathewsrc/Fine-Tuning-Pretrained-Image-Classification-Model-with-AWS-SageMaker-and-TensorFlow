import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
import argparse
import smdebug.tensorflow as smd

# Define image size
IMG_SIZE = (160, 160)

# Define image shape (3, 160, 160)
IMG_SHAPE = IMG_SIZE + (3,)


def net():
    """
    This function defines the neural network architecture for image classification.
    It uses the pre-trained MobileNetV2 model and adds data augmentation, dropout, and a dense layer to make predictions.

    Returns:
        model: The compiled Keras model.
    """
    # Define data augmentation pipeline
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ]
    )

    # Preprocess input images
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Create the base model from the pre-trained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    # Add a global average pooling layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Add a dense layer with a single output node to make predictions
    prediction_layer = tf.keras.layers.Dense(1)

    # Define inputs and outputs for the model
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    # Create the model and compile it
    model = tf.keras.Model(inputs, outputs)
    return model


def train_model(model, train_ds, val_ds, hook, args):
    """
    Train a machine learning model.

    Parameters:
    model (keras.models.Model): The model to be trained.
    train_ds (tf.data.Dataset): The dataset to use for training.
    val_ds (tf.data.Dataset): The dataset to use for validation.
    epochs (int): The number of epochs to train the model for.
    hook (smdebug.TensorFlowDebugHook): The hook to use for saving tensor values during training.

    """
    # Set the hook to training mode
    hook.set_mode(mode=smd.modes.TRAIN)

    # Train the model
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[hook])


def test(model, test_ds, hook, args):
    """
    Evaluate a machine learning model and save the results.

    Parameters:
    model (keras.models.Model): The model to be evaluated.
    test_ds (tf.data.Dataset): The dataset to use for testing.
    hook (smdebug.TensorFlowDebugHook): The hook to use for saving tensor values during evaluation.
    model_dir (str): The directory to save the model in.

    """
    # Set the hook to evaluation mode
    hook.set_mode(mode=smd.modes.EVAL)

    # Evaluate the model
    model.evaluate(test_ds, callbacks=[hook], batch_size=args.batch_size)

    # Set the version number
    version = "00000000"

    # Create a directory for the checkpoint
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Save the model
    model.save(ckpt_dir)


def main(args):

    # Load the training and testing datasets using image_dataset_from_directory method
    train_ds = image_dataset_from_directory(
        args.training, batch_size=args.batch_size, seed=123, image_size=IMG_SIZE
    )
    test_ds = image_dataset_from_directory(
        args.testing, batch_size=args.batch_size, seed=123, image_size=IMG_SIZE
    )

    # Split the training dataset into validation and training datasets
    val_batches = tf.data.experimental.cardinality(train_ds)
    train_ds = train_ds.take(val_batches // 5)
    val_ds = train_ds.skip(val_batches // 5)

    # Use prefetch to improve performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Use prefetch to improve performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Create model
    model = net()

    # Create hook
    hook = smd.KerasHook.create_from_json_file()

    # Create an optimizer
    optimizer = Adam(
        learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2
    )

    # Wrap the optimizer with wrap_optimizer so smdebug can find gradients to save
    optimizer = hook.wrap_optimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    train_model(model, train_ds, val_ds, hook, args)

    # Evaluate and save the model
    test(model, test_ds, hook, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 (Best Model)")

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)

    # Input data, output directory and model directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--out_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--testing", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    opt, _ = parser.parse_known_args()

    main(opt)
