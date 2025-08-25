from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, TimeDistributed
from tensorflow.keras.layers import Dropout, BatchNormalization, GaussianNoise

def create_model(sequence_length, image_height, image_width, num_classes):
    model = Sequential([
        TimeDistributed(GaussianNoise(0.01), input_shape=(sequence_length, image_height, image_width, 3)),
        TimeDistributed(Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((3, 3), strides=2)),

        TimeDistributed(Conv2D(256, (5, 5), padding='same', activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((3, 3), strides=2)),

        TimeDistributed(Conv2D(384, (3, 3), padding='same', activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Conv2D(384, (3, 3), padding='same', activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu')),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D((3, 3), strides=2)),

        TimeDistributed(Flatten()),
        Dense(4096, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(4096, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
