from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
from numpy.random import seed
from tensorflow import set_random_seed
import random


random.seed(1500)
seed(1822)
set_random_seed(1889)


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


DATA_FOLDER = 'proteins'
IMAGE_SIZE = [64, 64, 3]
BATCH_SIZE = 128
model = load_model(DATA_FOLDER + '/ckpt/5_0.7658.h5',
                   custom_objects={'precision_score': precision_score,
                                   'recall_score': recall_score})

datagen = ImageDataGenerator(preprocessing_function=None,
                             rescale=1.0/255.0)

val_df = pd.read_csv(DATA_FOLDER + '/crossval/test_5.csv')
labels_test = val_df['class']
val_df['class'] = val_df['class'].apply(str)

test_gen = datagen.flow_from_dataframe(dataframe=val_df,
                                       directory=None,  # df already has absolute paths
                                       x_col='image',
                                       y_col='class',
                                       target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                       interpolation='nearest',
                                       shuffle=False,
                                       class_mode='categorical',
                                       batch_size=BATCH_SIZE,
                                       color_mode='rgb')
preds = model.predict_generator(test_gen,
                                steps=test_gen.samples // BATCH_SIZE + 1,
                                workers=-1)
preds = np.argmax(preds, axis=1)
preds = np.eye(2)[preds]
labels_test = to_categorical(labels_test)
acc = accuracy_score(np.array(labels_test), preds, normalize=True)
print('Train acc:', acc)

