import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input, Lambda,
                                     MaxPooling2D, concatenate)
from tensorflow.keras.models import Model

import metrics

seed = 56


def model_generation(image_height= 256, image_width = 256, num_channels = 3, learning_rate = 0.001):
    print("Model generation...")

    inputs = Input((image_height, image_width, num_channels))
    s = Lambda(lambda x: x / 255) (inputs)

    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = BatchNormalization() (conv1)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = BatchNormalization() (conv1)
    pooling1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling1)
    conv2 = BatchNormalization() (conv2)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = BatchNormalization() (conv2)
    pooling2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling2)
    conv3 = BatchNormalization() (conv3)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = BatchNormalization() (conv3)
    pooling3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling3)
    conv4 = BatchNormalization() (conv4)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = BatchNormalization() (conv4)
    pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pooling4)
    conv5 = BatchNormalization() (conv5)
    conv5 = Dropout(0.3) (conv5)
    conv5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv5)
    conv5 = BatchNormalization() (conv5)


    upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample6 = concatenate([upsample6, conv4])
    conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample6)
    conv6 = BatchNormalization() (conv6)
    conv6 = Dropout(0.2) (conv6)
    conv6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv6)
    conv6 = BatchNormalization() (conv6)

    upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    upsample7 = concatenate([upsample7, conv3])
    conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample7)
    conv7 = BatchNormalization() (conv7)
    conv7 = Dropout(0.2) (conv7)
    conv7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv7)
    conv7 = BatchNormalization() (conv7)

    upsample8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
    upsample8 = concatenate([upsample8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample8)
    conv8 = BatchNormalization() (conv8)
    conv8 = Dropout(0.1) (conv8)
    conv8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv8)
    conv8 = BatchNormalization() (conv8)

    upsample9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
    upsample9 = concatenate([upsample9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (upsample9)
    conv9 = BatchNormalization() (conv9)
    conv9 = Dropout(0.1) (conv9)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv9)
    conv9 = BatchNormalization() (conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # print(model.summary())

    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=opt,
        loss=metrics.soft_dice_loss,
        metrics=[K.metrics.MeanIoU, metrics.precision, metrics.recall, metrics.f1_score])
    
    return model
