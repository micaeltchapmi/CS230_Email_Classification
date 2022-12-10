import os
from statistics import mean
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import count as count_func
import json
import torch
from collections import defaultdict

from PIL import Image
import io

from datetime import datetime

import pickle

#CIFAR 10
def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            Y = np.array(Y)
            return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        #break  #debug  
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

#Debug(overfit): replace num_training=[9000 to 49000] and vice versa and add/remove break in function above
def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=10000, show_sample=True):
    """
    Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
    """

    cifar10_dir = './data/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # subsample the data for validation set
    mask = range(num_training, num_training + num_val)
    X_val = X_train[mask]
    y_val = y_train[list(mask)]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    return X_train, y_train, X_val, y_val, X_test, y_test

def visualize_sample(X_train, y_train, classes, samples_per_class=7):
    """visualize some samples in the training datasets """
    num_classes = len(classes)
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y) # get all the indexes of cls
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs): # plot the image one by one
            plt_idx = i * num_classes + y + 1 # i*num_classes and y+1 determine the row and column respectively
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
    
def preprocessing_CIFAR10_data(args, X_train, y_train, X_val, y_val, X_test, y_test):
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)) # [49000, 3072]
    X_val = np.reshape(X_val, (X_val.shape[0], -1)) # [1000, 3072]
    X_test = np.reshape(X_test, (X_test.shape[0], -1)) # [10000, 3072]
    
    # Normalize the data: subtract the mean image
    """
    mean_image = np.mean(X_train, axis = 0)
    args.mean_image = mean_image
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    """

    # normalize in range -1 and 1
    X_train = (X_train - 127.5) / 127.5
    X_val = (X_val - 127.5) / 127.5
    X_test = (X_test - 127.5) / 127.5

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_img(args, path, method=0):
    img = np.load(path)
    return img


def save_prediction(args, image, rec, imgname, bid, img_id, epoch):
    W, H, C = image.shape
    #img = (image.reshape(-1) + args.mean_image).astype(np.uint8).reshape(W, H, C)
    #img_rec = np.transpose(((rec * 255.0).flatten() + args.mean_image).astype(np.uint8).reshape(C,W,H), (1,2,0))
    img = ((image.reshape(-1) * 127.5) + 127.5).astype(np.uint8).reshape(W, H, C)
    img_rec = np.transpose(((rec * 127.5).flatten() + 127.5).astype(np.uint8).reshape(C,W,H), (1,2,0))

    fig = plt.figure()
    rows, columns = 1, 2
    fig.add_subplot(rows, columns, 1)

    plot = plt.imshow(img)
    plot.axes.get_xaxis().set_visible(False)
    plot.axes.get_yaxis().set_visible(False)

    fig.add_subplot(rows, columns, 2)
    plot2 = plt.imshow(img_rec)
    plot2.axes.get_xaxis().set_visible(False)
    plot2.axes.get_yaxis().set_visible(False)

    plt.title(imgname)
    fig.set_size_inches(np.array(fig.get_size_inches()))
    fig.tight_layout()
    fig.patch.set_alpha(1)
    #plt.show(block=False)

    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, imgname + str(img_id) + '.png'), facecolor=fig.get_facecolor())
    plt.close()

def show_image(image, label=None, title=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.imshow(image)
    plt.title(title + "_ID: " + str(label))
    plt.show()
    

if __name__ == "__main__":
    pass

