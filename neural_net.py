import numpy as np
import pandas as pd
import random

from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import backend as K
import cv2 as cv

# random.seed(29)
N_CLASSES = 3
BATCH_SIZE = 64


def precision_score(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_score(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def get_mobilenet_model():
    mobile = MobileNetV2(weights=None,
                         input_tensor=Input(shape=(178, 178, 1)))
    print(mobile.summary())
    x = mobile.layers[-2].output  # -4 se for mobilenet v1
    x = Dense(128, activation='relu')(x)
    pred = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', precision_score, recall_score])
    return model


def get_mlp_model():

    input_layer = Input(shape=[None, None, 3])

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(input=input_layer, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', precision_score, recall_score])

    return model


def get_model():
    input_layer = Input(shape=[None, None, 3])

    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(input=input_layer, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', precision_score, recall_score])

    return model


model = get_model()
print(model.summary())

# image = cv.imread('img/1/0a4ecdf517.png')
# print(np.shape(image))


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)

datagen = ImageDataGenerator(preprocessing_function=None,
                             rescale=1.0/255.0,
                             # horizontal_flip=True,
                             # vertical_flip=True,
                             validation_split=0.2)

train_gen = datagen.flow_from_directory('imdb/img/',
                                        batch_size=BATCH_SIZE,
                                        subset='training',
                                        color_mode='rgb')

test_gen = datagen.flow_from_directory('imdb/img/',
                                       batch_size=BATCH_SIZE,
                                       subset='validation',
                                       color_mode='rgb')

model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE + 1,
                    validation_data=test_gen,
                    validation_steps=test_gen.samples // BATCH_SIZE + 1,
                    epochs=100,
                    verbose=1, callbacks=[reduce_lr],
                    workers=-1)

model.save('saved_model.h5')


