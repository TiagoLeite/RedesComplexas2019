import numpy as np
import pandas as pd
import random
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import backend as K
import cv2 as cv
from keras.optimizers import Adam, RMSprop
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold
from glob import glob
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.utils import to_categorical

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


def get_autoencoder():
    input_layer = Input(shape=[128, 128, 3])
    # x = GaussianNoise(0.25)(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    encoder = Dense(units=256, activation='relu', name='encoder')(x)

    x = Dense(units=2048, activation='relu')(encoder)
    x = Reshape(target_shape=[4, 4, 128])(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # x = UpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    output = Conv2D(3, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    # output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(input=input_layer, output=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def k_folds(k, data_folder):
    X = glob(data_folder + '/img/*/*')
    y = [int(image.split('/')[-2]) for image in X]
    skfold = StratifiedKFold(n_splits=k, random_state=1899)
    split_count = 0
    for train_index, test_index in skfold.split(X, y):
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        x_train = [X[i] for i in train_index]
        x_test = [X[i] for i in test_index]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        dataframe_train = pd.DataFrame(data={'image': x_train, 'class': y_train})
        dataframe_test = pd.DataFrame(data={'image': x_test, 'class': y_test})
        dataframe_train.to_csv(data_folder + '/crossval/train_' + str(split_count) + '.csv', index=False)
        dataframe_test.to_csv(data_folder + '/crossval/test_' + str(split_count) + '.csv', index=False)
        split_count += 1


def plot_pca(x_data, y_data, x_test, y_test, fold):
    pca = PCA(n_components=2)
    comps_train = pca.fit_transform(X=x_data)
    comps_test = pca.fit_transform(X=x_test)
    pl.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    color_map = [colors[y_data[k]] for k in range(len(y_data))]
    color_map_test = [colors[y_test[k]] for k in range(len(y_test))]
    plt.scatter(comps_train[:, 0], comps_train[:, 1], c=color_map, marker='o')
    plt.scatter(comps_test[:, 0], comps_test[:, 1], facecolors='w', edgecolors=color_map_test, marker='o')
    plt.title('Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(DATA_FOLDER + '/scatter' + str(fold) + '.png')


def get_model():
    input_layer = Input(shape=IMAGE_SIZE)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu', name='embedding_layer')(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(input=input_layer, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
                  metrics=['accuracy', precision_score, recall_score])
    return model


def train_classification():
    results = list()
    for k in range(K_FOLDS):
        print("==================== Fold ", k, "====================")
        print("Datasets:", 'train_' + str(k) + '.csv', 'test_' + str(k) + '.csv')
        print('Training for', DATA_FOLDER)
        model = get_model()
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                      mode='max',
                                      verbose=2,
                                      patience=10,
                                      min_lr=1e-6)
        datagen = ImageDataGenerator(preprocessing_function=None,
                                     rescale=1.0/255.0)
                                     #horizontal_flip=True,
                                     #vertical_flip=True)

        train_df = pd.read_csv(DATA_FOLDER + '/crossval/train_' + str(k) + '.csv')
        val_df = pd.read_csv(DATA_FOLDER + '/crossval/test_' + str(k) + '.csv')

        labels_train = train_df['class']
        labels_test = val_df['class']

        train_df['class'] = train_df['class'].apply(str)
        val_df['class'] = val_df['class'].apply(str)

        train_gen = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory=None,  # df already has absolute paths
                                                x_col='image',
                                                y_col='class',
                                                shuffle=False,
                                                interpolation='nearest',
                                                target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                class_mode='categorical',
                                                batch_size=BATCH_SIZE,
                                                color_mode='rgb')

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
        checkpoint = ModelCheckpoint(DATA_FOLDER+'/ckpt/'+str(k)+'_{val_acc:.4f}.h5',
                                     monitor='val_acc',
                                     save_best_only=True,
                                     verbose=1,
                                     save_weights_only=False,
                                     mode='max', period=1)
        csv_logger = CSVLogger(DATA_FOLDER + '/log/training' + str(k) + '.log')
        model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE + 1,
                            validation_data=test_gen,
                            validation_steps=test_gen.samples // BATCH_SIZE + 1,
                            epochs=EPOCHS,
                            verbose=2,
                            callbacks=[csv_logger, checkpoint],
                            workers=-1)

        preds = model.predict_generator(test_gen, workers=-1)
        preds = np.argmax(preds, axis=1)
        preds_cat = to_categorical(preds)
        labels_test_cat = to_categorical(labels_test)
        acc = accuracy_score(np.array(labels_test_cat), preds_cat)
        print('Train acc:', acc)

        code_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
        code_train = np.asarray(code_model.predict_generator(train_gen))
        code_test = np.asarray(code_model.predict_generator(test_gen))
        scaler = StandardScaler()
        code_train = scaler.fit_transform(code_train)
        code_test = scaler.fit_transform(code_test)
        # code_train = normalize(code_train, axis=1)
        # code_test = normalize(code_test, axis=1)
        '''print("Training data size = ", code_train.shape)
        print("Testing data size = ", code_test.shape)
        print('Treinando SVM...')
        clf = svm.LinearSVC(C=2, max_iter=100000)
        clf.fit(code_train, labels_train)
        print('Calculando score...')
        preds = clf.predict(code_test)
        np.save(DATA_FOLDER + '/preds_' + str(k) + '.npy', preds)
        np.save(DATA_FOLDER + '/labels_' + str(k) + '.npy', labels_test)
        np.save(DATA_FOLDER + '/encoding_' + str(k) + '.npy', code_test)
        report = clf.score(code_test, labels_test)
        results.append(report)
        print('\nScore: ', report)'''
        plot_pca(code_train, labels_train, code_test, preds, k)

    print('Final results:', results)


def train_autoencoder():
    results = list()
    for k in range(K_FOLDS):
        print("==================== Fold ", k, "====================")
        print("Datasets:", 'train_' + str(k) + '.csv', 'test_' + str(k) + '.csv')
        print('Training for', DATA_FOLDER)
        model = get_autoencoder()
        if k == 0:
            print(model.summary())
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      mode='max',
                                      verbose=2,
                                      patience=10,
                                      min_lr=1e-6)
        datagen = ImageDataGenerator(preprocessing_function=None,
                                     rescale=1.0 / 255.0,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        train_df = pd.read_csv(DATA_FOLDER + '/crossval/train_' + str(k) + '.csv')
        val_df = pd.read_csv(DATA_FOLDER + '/crossval/test_' + str(k) + '.csv')

        labels_train = train_df['class']
        labels_test = val_df['class']

        train_df['class'] = train_df['class'].apply(str)
        val_df['class'] = val_df['class'].apply(str)

        train_gen = datagen.flow_from_dataframe(dataframe=train_df,
                                                directory=None,  # df already has absolute paths
                                                x_col='image',
                                                y_col='class',
                                                shuffle=False,
                                                interpolation='nearest',
                                                target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                class_mode='input',
                                                batch_size=BATCH_SIZE,
                                                color_mode='rgb')

        test_gen = datagen.flow_from_dataframe(dataframe=val_df,
                                               directory=None,  # df already has absolute paths
                                               x_col='image',
                                               y_col='class',
                                               target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                               interpolation='nearest',
                                               shuffle=False,
                                               class_mode='input',
                                               batch_size=BATCH_SIZE,
                                               color_mode='rgb')

        csv_logger = CSVLogger(DATA_FOLDER + '/log/training' + str(k) + '.log')
        model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE + 1,
                            validation_data=test_gen,
                            validation_steps=test_gen.samples // BATCH_SIZE + 1,
                            epochs=EPOCHS,
                            shuffle=False,
                            verbose=1,
                            callbacks=[csv_logger],
                            workers=-1)

        code_model = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
        code_train = np.asarray(code_model.predict_generator(train_gen))
        code_test = np.asarray(code_model.predict_generator(test_gen))
        scaler = StandardScaler()
        code_train = scaler.fit_transform(code_train)
        code_test = scaler.fit_transform(code_test)
        # code_train = normalize(code_train, axis=1)
        # code_test = normalize(code_test, axis=1)
        print("Training data size = ", code_train.shape)
        print("Testing data size = ", code_test.shape)
        print('Treinando SVM...')
        clf = svm.LinearSVC(C=2, max_iter=100000)
        clf.fit(code_train, labels_train)
        print('Calculando score...')
        preds = clf.predict(code_test)
        np.save(DATA_FOLDER + '/preds_' + str(k) + '.npy', preds)
        np.save(DATA_FOLDER + '/labels_' + str(k) + '.npy', labels_test)
        np.save(DATA_FOLDER + '/encoding_' + str(k) + '.npy', code_test)
        report = clf.score(code_test, labels_test)
        results.append(report)
        print('\nScore: ', report)
        plot_pca(code_train, labels_train, k)
    print(results)


DATA_FOLDER = input('Dataset: proteins, imdb, synthetic: ')
N_CLASSES = int(input('Number of classes: '))
IMAGE_SIZE = [int(input('Image shape:')), int(input()), int(input())]
print('Image shape:', IMAGE_SIZE)
# DATA_FOLDER = 'proteins'
# N_CLASSES = 2
BATCH_SIZE = 128
K_FOLDS = 10
EPOCHS = 100

k_folds(K_FOLDS, DATA_FOLDER)
train_classification()
# train_autoencoder()
