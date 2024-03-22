from tensorflow.keras import models, layers

def create_cnn(
    kernel_sizes=[3, 3],
    num_filters=[32, 64],
    filter_stride=1,


    pooling_size=2,
    pooling_stride=2,

    fc_sizes=[128, 64],

    image_size=32,
    num_input_channels=3,
    num_classes=10,

    dropout_prob=0.1,

    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
):

    model = models.Sequential()

    # First convolutional layer with specified input shape
    model.add(layers.Conv2D(num_filters[0], kernel_size=kernel_sizes[0], strides=filter_stride, activation='relu', input_shape=(image_size, image_size, num_input_channels)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pooling_size, strides=pooling_stride))

    # Subsequent convolutional layers
    for kernel_size, num_filter in zip(kernel_sizes[1:], num_filters[1:]):
        model.add(layers.Conv2D(num_filter, kernel_size=kernel_size, strides=filter_stride, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=pooling_size, strides=pooling_stride))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected layers
    for fc_size in fc_sizes:
        model.add(layers.Dense(fc_size, activation='relu'))
        model.add(layers.Dropout(dropout_prob))

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model