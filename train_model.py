import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import OneHotEncoder

def fcn(n_classes):
    model = keras.Sequential([
        layers.Input(shape=(None, None, 3)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPool2D((3,3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPool2D((3,3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPool2D((3,3)),
        layers.Conv2D(128, (7, 7), activation='relu'),
        layers.Conv2D(128, (1, 1), activation='relu'),
        layers.Conv2D(n_classes, (1, 1), activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    return model

def get_data(indir='./data/images/train/'):
    labels = []
    images = []
    for label in os.listdir(indir):
        subdir = os.path.join(indir, label)
        for filename in os.listdir(subdir):
            fullname = os.path.join(subdir, filename)
            images.append(cv2.imread(fullname))
            labels.append(label)

    return np.array(images), np.array(labels).reshape(-1, 1)

def train_test_index_split(X, y, ratio=0.2):
    train_index = []
    test_index = []
    for ycur in np.unique(y[:, 0]):
        index_match = np.argwhere(y.ravel() == ycur).flatten()
        cutoff = int(index_match.shape[0] * ratio)
        left, right = np.split(index_match, [cutoff])
        test_index.append(left)
        train_index.append(right)

    train_index = np.concatenate(train_index)
    test_index = np.concatenate(test_index)
    
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)

    return train_index, test_index

def multilabel_statistics(yact, ypred, labels):
    n_labels = len(labels)
    # actual, pred
    obs = np.zeros(shape=(n_labels + 1, n_labels + 1))
    for act, pred in zip(yact, ypred):
        act_index = np.argwhere(act == 1).flatten()
        likely = np.argwhere(pred >= 0.5).flatten() 
        if likely.shape[0] == 0:
            obs[act_index, n_labels] += 1
        else:
            for l in likely:
                obs[act_index, l] += 1

    for row, label in zip(obs, labels):
        print (f'{label:20} {row} ')

    return obs



if __name__ == '__main__':
    images, labels = get_data()

    # Encode labels for model
    le = OneHotEncoder(sparse=False).fit(labels)
    unique_labels = le.categories_[0]
    n_labels = len(unique_labels)

    # Samples per categories
    print(unique_labels, n_labels)

    # Create train & test indices
    train_index, test_index = train_test_index_split(images, labels, ratio=0.2)

    # Prep data & label for training
    Xtrain = images[train_index, :, :, :] / 255.
    ytrain = le.transform(labels[train_index, :]).reshape(-1, 1, 1, n_labels)

    model = fcn(n_labels)
    history = model.fit(Xtrain, ytrain, epochs=10, batch_size=32)

    # Prep data & label for testing
    Xtest = images[test_index, :, :, :] / 255.
    ytest = le.transform(labels[test_index, :])

    # Make predictions
    ypred = model.predict(Xtest).reshape(-1, n_labels)    

    multilabel_statistics(ytest, ypred, unique_labels)

    # Save model
    model.save('./data/model/mymodel.h5')
    # Save label
    with open('data/model/mymodel.txt', 'w') as f:
        [f.write(f'{i}\n') for i in unique_labels]
