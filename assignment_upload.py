import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

import pickle
import time

import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from numpy import random as rng
import tensorflow as tf
import tensorflow_datasets as tfds


def convert_images_to_pickle(image_folder_path, pickle_file_path):
    X = []
    labels = {}
    no = 0

    for language in os.listdir(image_folder_path):
        labels[language] = [no, None]
        alphabet_path = os.path.join(image_folder_path, language)

        for letter in os.listdir(alphabet_path):
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)

            try:
                X.append(np.stack(category_images))

            except ValueError as e:
                print(e)

            no += 1
            labels[language][1] = no - 1

    X = np.stack(X)
    with open(pickle_file_path, 'wb') as filename:
        pickle.dump((X, labels), filename)
    return


def euclidean_distance(tensors):
    return K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True))


def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    m = 1

    twice_loss = (1 - y_true) * tf.square(y_pred) + y_true * tf.square(tf.maximum(0., m - y_pred))

    loss = 0.5 * tf.reduce_mean(twice_loss)
    return loss


def triplet_loss(tensors):
    anchor_output = tensors[0]
    positive_output = tensors[1]
    negative_output = tensors[2]

    alpha = 0.5
    d_positive = K.sum(K.square(anchor_output - positive_output), axis=1, keepdims=True)
    d_negative = K.sum(K.square(anchor_output - negative_output), axis=1, keepdims=True)

    loss = K.maximum(0., d_positive - d_negative + alpha)

    return loss


def test_different_label_contrastive_loss():
    left_output = tf.Variable(K.random_normal([1, 4096], mean=0, stddev=0.01))
    right_output = tf.Variable(K.random_normal([1, 4096], mean=1, stddev=0.01))
    y_true = 1

    l1_distance = Lambda(euclidean_distance, output_shape=output_shape)([left_output, right_output])
    y_pred = Dense(1, activation='sigmoid')(l1_distance)
    loss = contrastive_loss(y_true, y_pred)
    print(loss)


def test_same_label_contrastive_loss():
    left_output = tf.Variable(K.random_normal([1, 4096], mean=0, stddev=0.001))
    right_output = tf.Variable(K.random_normal([1, 4096], mean=0, stddev=0.001))
    y_true = 0

    l1_distance = Lambda(euclidean_distance, output_shape=output_shape)([left_output, right_output])
    y_pred = Dense(1, activation='sigmoid')(l1_distance)
    loss = contrastive_loss(y_true, y_pred)
    print(loss)


def test_triplet_loss():
    anchor_output = tf.Variable(K.random_normal([1, 4096], mean=0, stddev=0.001))
    positive_output = tf.Variable(K.random_normal([1, 4096], mean=0, stddev=0.001))
    negative_output = tf.Variable(K.random_normal([1, 4096], mean=1, stddev=0.001))

    l1_distance = Lambda(triplet_loss, output_shape=output_shape)(
        [anchor_output, positive_output, negative_output])
    y_pred = Dense(1, activation='sigmoid')(l1_distance)
    print(l1_distance)
    print(y_pred)


class DataGenerator:

    def __init__(self, batch_size):
        self.X_train, self.y_train = self.__initialize_data(dataset_name='omniglot', split='train')
        self.X_test, self.y_test = self.__initialize_data(dataset_name='omniglot', split='test')
        self.num_classes = 50

        self.batch_size = batch_size

    def __initialize_data(self, dataset_name='omniglot', split='train'):
        ds = tfds.load(dataset_name, split=split, with_info=False, download=True)
        X, y = [], []
        for data in tfds.as_numpy(ds):
            image = data.get('image')
            label = data.get('alphabet')

            X.append(image)
            y.append(label)

        X = np.array(X)
        y = np.array(y).astype('float32')
        return X, y

    def get_image(self, label, test=False):
        """Choose an image from our training or test data with the
        given label."""
        if test:
            y = self.y_test
            X = self.X_test
        else:
            y = self.y_train
            X = self.X_train
        idx = np.random.randint(len(y))
        while y[idx] != label:
            # keep searching randomly!
            idx = np.random.randint(len(y))
        return X[idx]

    def get_triplet(self, test=False):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""
        n = a = np.random.randint(10)
        while n == a:
            # keep searching randomly!
            n = np.random.randint(10)
        a, p = self.get_image(a, test), self.get_image(a, test)
        n = self.get_image(n, test)
        return a, p, n

    def generate_triplets(self, batch_size, test=False):
        """Generate an un-ending stream (ie a generator) of triplets for
        training or test."""
        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batch_size):
                a, p, n = self.get_triplet(test)
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)

            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            # a "dummy" label which will come in to our identity loss
            # function below as y_true. We'll ignore it.
            label = np.ones(batch_size)
            yield [A, P, N], label

    def get_batch(self, test=False):
        if test:
            X = self.X_test
            y = self.y_test
        else:
            X = self.X_train
            y = self.y_train

        batch_size = self.batch_size

        category = dict()
        for i in range(0,len(y)):
            label = y[i]
            image_list = category.get(label)
            if image_list is None:
                image_list = []

            image_list.append(i)
            category[label] = image_list

        n_examples, w, h, c = X.shape
        cat = rng.choice(list(category.keys()), size=batch_size, replace=True)
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        pairs = [np.zeros((batch_size, w,h,c)) for _ in range(2)]
        for i in range(batch_size):
            image_id_list_for_category = category.get(cat[i])
            ex_no = rng.randint(len(image_id_list_for_category))
            pairs[0][i] = X[image_id_list_for_category[ex_no],:, :, :]

            if i <= batch_size // 2:
                cat2 = cat[i]
            else:
                key_list = list(category.keys())
                key_list.remove(cat[i])
                cat2 = rng.choice(key_list, size=1, replace=False)[0]

            image_id_list_for_category = category.get(cat2)
            ex_no2 = rng.randint(len(image_id_list_for_category))
            pairs[1][i] = X[image_id_list_for_category[ex_no2], :, :, :]

        return pairs, targets


def average_loss(y_true, y_pred):
    return K.mean(y_pred)


def get_base_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer='l2'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer='l2'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer='l2'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer='l2'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer='l2'))

    return model


def siamese_model_triplet_loss(input_shape):
    learning_rate = 0.01
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    input_3 = Input(input_shape)

    base_model = get_base_model(input_shape)

    anchor = base_model(input_1)
    positive = base_model(input_2)
    negative = base_model(input_3)

    distance = Lambda(triplet_loss)([anchor, positive, negative])
    # prediction = Dense(1, activation='sigmoid')(distance)
    prediction = distance
    model = Model(inputs=[input_1, input_2, input_3], outputs=prediction)
    model.compile(loss=average_loss, optimizer=Adam(lr=learning_rate))

    model.summary()

    return model


def testing():
    input_shape = (105, 105, 3)
    learning_rate = 0.00006
    batch_size = 32
    epoches = 20
    data_generator = DataGenerator(batch_size)
    train_data_generator = data_generator.generate_triplets(batch_size)
    test_data_generator = data_generator.generate_triplets(batch_size, True)

    model = siamese_model_triplet_loss(input_shape)
    model.summary()
    history = model.fit_generator(generator=train_data_generator,
                                  validation_data=test_data_generator,
                                  epochs=epoches,
                                  verbose=2, steps_per_epoch=1,
                                  validation_steps=1)
    model.save_weights('triplet_loss_model.hdf5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Losses', size=20)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def generate_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = get_base_model(input_shape)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # l1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # l1_distance = l1_layer([encoded_l, encoded_r])
    # l1_distance = euclidean_distance([encoded_l, encoded_r])
    distance = Lambda(euclidean_distance, output_shape=output_shape)([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid')(distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net


def load_data_from_pickle_file(file_path):
    X = None
    classses = None

    with open(file_path, "rb") as f:
        (X, classes) = pickle.load(f)
    return X, classes


def get_batch(X, batch_size):
    n_classes, n_examples, w, h = X.shape
    cat = rng.choice(n_classes, size=batch_size, replace=False)
    targets = np.zeros((batch_size,))
    targets[batch_size // 2:] = 1
    pairs = [np.zeros((batch_size, h, w, 1)) for _ in range(2)]
    for i in range(batch_size):
        ex_no = rng.randint(n_examples)
        pairs[0][i, :, :, :] = X[cat[i], ex_no, :, :].reshape(w, h, 1)
        cat2 = 0
        if i >= batch_size // 2:
            cat2 = cat[i]
        else:
            cat2 = (cat[i] + rng.randint(1, n_classes)) % n_classes
        ex_no2 = rng.randint(n_examples)
        pairs[1][i, :, :, :] = X[cat2, ex_no2, :, :].reshape(w, h, 1)
    return pairs, targets


def step_2_test_contrastive_loss_function():
    pass


def step_3_test_triplet_loss_function():
    pass


def create_pairs(X, Y, num_classes):
    pairs, labels = [], []

    category = dict()
    for i in range(0, len(Y)):
        label = Y[i]
        image_list = category.get(label)
        if image_list is None:
            image_list = []

        image_list.append(i)
        category[label] = image_list

    class_ids = list(category.keys())
    class_ids.sort()
    class_idx = [None for i in class_ids]

    j = 0
    for id in class_ids:
        class_idx[j]=category.get(id)
        j +=1

    num_classes = len(class_ids)
    # index of images in X and Y for each class
    # class_idx = [np.where(Y == i)[0] for i in range(num_classes)]
    # The minimum number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1

    for c in range(num_classes):
        for n in range(min_images):
            # create positive pair
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n + 1]]
            pairs.append((img1, img2))
            labels.append(1)

            # create negative pair
            # first, create list of classes that are different from the current class
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            # select a random class from the negative list.
            # this class will be used to form the negative pair
            neg_c = random.sample(neg_list, 1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1, img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)


def step_4_train_model():
    input_shape = (105, 105, 3)
    learning_rate = 0.00006
    batch_size = 32
    epoches = 2000

    training_time = []
    training_errors = []
    validation_errors = []

    siamese_model = generate_siamese_model(input_shape)

    siamese_model.summary()

    optimizer = Adam(lr=learning_rate)
    # siamese_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    siamese_model.compile(loss=contrastive_loss, optimizer=optimizer)
    data_generator = DataGenerator(batch_size)
    (X_train, Y_train) = data_generator.X_train, data_generator.y_train
    (X_test, Y_test) = data_generator.X_test, data_generator.y_test
    num_classes = len(np.unique(Y_train))
    training_pairs, training_labels =create_pairs(X_train, Y_train, num_classes=num_classes)
    training_labels.astype(np.float32)
    training_pairs.astype(np.float32)
    siamese_model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
              batch_size=128,
              epochs=10)

    # Xtrain, y_test = load_data_from_pickle_file("./train.pickle")
    # Xtest, y_test = load_data_from_pickle_file("./val.pickle")

    # for epoch in range(1, epoches):
    #     (inputs, targets) = get_batch(Xtrain, batch_size)
    #     training_error = siamese_model.train_on_batch(inputs, targets)
    #     (inputs, targets) = get_batch(Xtest, batch_size)
    #     validation_error = siamese_model.evaluate(inputs, targets)
    #     current_time = time.time()
    #
    #     training_errors.append(training_error)
    #     validation_errors.append(validation_error)
    #     training_time.append(current_time)
    #
    # epochs_range = [i for i in range(1, len(training_time) + 1)]
    #
    # plt.figure(figsize=(5, 5))
    #
    # plt.plot(epochs_range, training_errors, label='Training Error')
    # plt.plot(epochs_range, validation_errors, label='Validation Error')
    # plt.legend(loc="best")
    # plt.xlabel('Iterations')
    # plt.ylabel('Error')
    # plt.title('Training and Validation Error')
    #
    # plt.tight_layout()
    # plt.show()


# step_1_generate_train_and_test_data()
# step_2_test_contrastive_loss_function()
# step_3_test_triplet_loss_function()
step_4_train_model()
# testing()
# test_triplet_loss()
# step_5()
