import numpy as np
import pandas as pd
import random
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold
from glob import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import matplotlib.patches as mpatches
from keras.models import load_model
from sklearn.metrics import accuracy_score

random.seed(1500)
seed(1822)
set_random_seed(1889)


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


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
    pl.figure()

    colors = np.array(
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
         '#17becf'])

    color_map = [colors[y_data[k]] for k in range(len(y_data))]

    plt.scatter(comps_train[:, 0], comps_train[:, 1], c=color_map,
                marker='o')

    all_classes = list()

    for k in range(N_CLASSES):
        clazz = mpatches.Patch(color=colors[k], label='classe ' + str(k + 1))
        all_classes.append(clazz)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(handles=all_classes)
    plt.savefig(DATA_FOLDER + '/scatter_' + str(fold) + '.png')
    plt.close()


def get_model():
    input_layer = Input(shape=IMAGE_SIZE)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu', name='embedding_layer')(x)
    x = Dropout(0.5)(x)
    output = Dense(N_CLASSES, activation='softmax')(x)
    model = Model(input=input_layer, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
                  metrics=['accuracy', precision, recall, f_score])
    return model


def train_classification():

    for k in range(K_FOLDS):
        print("==================== Fold ", k, "====================")
        print("Datasets:", 'train_' + str(k) + '.csv', 'test_' + str(k) + '.csv')
        print('Training for', DATA_FOLDER)

        model = get_model()
        print(model.summary())

        early_stopping = EarlyStopping(monitor='val_acc',
                                       mode='max',
                                       verbose=2,
                                       min_delta=0.01,
                                       patience=32)

        checkpoint = ModelCheckpoint(DATA_FOLDER + '/ckpt/best_model_'+str(k)+'.h5',
                                     monitor='val_acc',
                                     save_best_only=True,
                                     verbose=2,
                                     save_weights_only=False,
                                     mode='max', period=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                      patience=10, verbose=2, mode='min',
                                      min_delta=0.01, min_lr=1e-6)

        datagen = ImageDataGenerator(preprocessing_function=None,
                                     rescale=1.0/255.0)

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

        csv_logger = CSVLogger(DATA_FOLDER + '/log/training' + str(k) + '.log')
        history = model.fit_generator(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE + 1,
                                      validation_data=test_gen,
                                      validation_steps=test_gen.samples // BATCH_SIZE + 1,
                                      epochs=EPOCHS,
                                      verbose=2,
                                      callbacks=[csv_logger, checkpoint, early_stopping, reduce_lr],
                                      workers=-1)

        pl.figure()
        plt.plot(history.history['loss'], label='treino')
        plt.plot(history.history['val_loss'], label='teste')
        plt.xlabel('época')
        plt.ylabel('custo')
        plt.legend()
        plt.show()
        plt.savefig(DATA_FOLDER + '/history_loss_' + str(k) + '.png')

        pl.figure()
        plt.plot(history.history['acc'], label='treino')
        plt.plot(history.history['val_acc'], label='teste')
        plt.legend()
        plt.xlabel('época')
        plt.ylabel('acurácia')
        plt.show()
        plt.savefig(DATA_FOLDER + '/history_acc_' + str(k) + '.png')

        plt.close()

        model = load_model(DATA_FOLDER + '/ckpt/best_model_'+str(k)+'.h5',
                           custom_objects={'recall': recall,
                                           'precision': precision,
                                           'f_score': f_score})

        preds = model.predict_generator(test_gen, workers=-1)
        preds = np.argmax(preds, axis=1)
        preds_cat = to_categorical(preds)
        labels_test_cat = to_categorical(labels_test)
        acc = accuracy_score(np.array(labels_test_cat), preds_cat)
        print('Best train acc:', acc)

        code_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
        code_train = np.asarray(code_model.predict_generator(train_gen))
        code_test = np.asarray(code_model.predict_generator(test_gen))
        scaler = StandardScaler()
        code_train = scaler.fit_transform(code_train)
        code_test = scaler.fit_transform(code_test)
        plot_pca(code_train, labels_train, code_test, preds, k)


def evaluate(DATA_FOLDER, k):

    datagen = ImageDataGenerator(preprocessing_function=None,
                                 rescale=1.0 / 255.0)

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

    model = load_model(DATA_FOLDER + '/ckpt/best_model_' + str(k) + '.h5',
                       custom_objects={'recall': recall,
                                       'precision': precision,
                                       'f_score': f_score})

    preds = model.predict_generator(test_gen, workers=-1)
    preds = np.argmax(preds, axis=1)
    preds_cat = to_categorical(preds)
    labels_test_cat = to_categorical(labels_test)
    acc = accuracy_score(np.array(labels_test_cat), preds_cat)
    print('Best train acc:', acc)

    code_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
    code_train = np.asarray(code_model.predict_generator(train_gen))
    code_test = np.asarray(code_model.predict_generator(test_gen))
    scaler = StandardScaler()
    code_train = scaler.fit_transform(code_train)
    code_test = scaler.fit_transform(code_test)
    plot_pca(code_train, labels_train, code_test, preds, k)


DATA_FOLDER = input('Dataset: proteins, imdb, synthetic, DD: ')
N_CLASSES = int(input('Number of classes: '))
IMAGE_SIZE = [int(input('Image shape:')), int(input()), int(input())]
print('Image shape:', IMAGE_SIZE)
BATCH_SIZE = 128
K_FOLDS = 10
EPOCHS = 300

k_folds(K_FOLDS, DATA_FOLDER)
train_classification()

# evaluate('synthetic', 0)
