import tensorflow as tf
import os
import argparse


# Define image size
IMG_SIZE = (160, 160)

# Define image shape
IMG_SHAPE = IMG_SIZE + (3,)


def train(model, args, train_ds, test_ds):
    """
    Trains a model and saves it.

    Parameters:
        model (tf.keras.Model): The model to be trained.
        args (namespace): Arguments for the training process, including:
            - epochs (int): The number of training epochs.
            - learning_rate (float): The learning rate for the Adam optimizer.
            - beta_1 (float): The beta_1 parameter for the Adam optimizer.
            - beta_2 (float): The beta_2 parameter for the Adam optimizer.
            - model_dir (str): The directory where the trained model will be saved.
        train_ds (tf.data.Dataset): The training dataset.
        test_ds (tf.data.Dataset): The test dataset.

    Returns:
        None
    """
    # Set up the Adam optimizer with specified learning rate, beta_1, and beta_2
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2
    )

    # Compile the model
    model.compile()

    # Define the loss function as Binary Crossentropy
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Create metrics to track the loss and accuracy during training and testing
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        # Use GradientTape to track the gradients
        with tf.GradientTape() as tape:
            # Make predictions on the current batch of data
            predictions = model(images, training=True)
            # Calculate the loss using the loss function
            loss = loss_fn(labels, predictions)
        # Calculate the gradients
        grad = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to the model's variables
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # Update the training loss and accuracy metrics
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # Use the model to make predictions on the input images
        predictions = model(images, training=False)
        # Calculate the loss of the model's predictions compared to the actual labels
        t_loss = loss_fn(labels, predictions)
        # Update the test loss metric with the calculated loss
        test_loss(t_loss)
        # Update the test accuracy metric with the accuracy of the model's predictions
        test_accuracy(labels, predictions)
        # Return nothing
        return

    # Print message indicating the start of the training process
    print("Training starts ...")

    # Loop over the number of defined epochs
    for epoch in range(args.epochs):
        # Reset the states of the train loss and accuracy metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        # Reset the states of the test loss and accuracy metrics
        test_loss.reset_states()
        test_accuracy.reset_states()

        # Loop over the batches of data in the training dataset
        for batch, (images, labels) in enumerate(train_ds):
            # Run a training step on the current batch of data
            train_step(images, labels)
            # Calculate the progress of the training process and print it
            progress = 100.0 * batch / len(train_ds)
            print("Training progress: {:.0f}%".format(progress))
        # Print message indicating the end of the current epoch
        print("Training finished!")

        # Print the current epoch number, the train loss, and the train accuracy
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result().numpy()}, "
            f"Accuracy: {train_accuracy.result().numpy()}, "
        )

        # Loop over the batches of data in the test dataset
        for images, labels in test_ds:
            # Run a test step on the current batch of data
            test_step(images, labels)

        # Print the test loss and accuracy after the current epoch
        print(f"Test Loss: {test_loss.result().numpy()}")
        print(f"Test Accuracy: {test_accuracy.result().numpy()}")

    # Set the version of the model to be saved
    # version = "00000000"
    # Define the directory where the model will be saved
    # model_dir = os.path.join(args.model_dir, version)
    # If the directory does not already exist, create it
    # if not os.path.exists(model_dir):
    # os.makedirs(model_dir)
    # Save the model in the defined directory
    # model.save(model_dir)


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
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    # Create the model and compile it
    model = tf.keras.Model(inputs, outputs)
    return model


def main(args):
    # Load the training and testing datasets using image_dataset_from_directory method
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.training, batch_size=args.batch_size, seed=123, image_size=IMG_SIZE
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.testing, batch_size=args.batch_size, seed=123, image_size=IMG_SIZE
    )

    # Use prefetch to improve performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Create model
    model = net()

    # Train model
    train(model, args, train_ds, test_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 (HPO)")

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)

    # Input data and model directories
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--testing", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    opt, _ = parser.parse_known_args()

    main(opt)
