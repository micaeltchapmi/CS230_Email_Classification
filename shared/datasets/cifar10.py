# --------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# --------------------------------------------------------
import random
import os, sys
import argparse
import numpy as np
import cv2
import glob
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataprocess import DataProcess, get_while_running, kill_data_processes
from data_utils import load_img, get_CIFAR10_data, preprocessing_CIFAR10_data, load_CIFAR10, visualize_sample
sys.path.insert(0, './')
from data_process import show_image


class CIFAR10DataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """CDI dataloader.
        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """

        # Load Cifar 10 dataset
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
        #visualize_sample(X_train_raw, y_train_raw, classes)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_CIFAR10_data(args, X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)

        #debug
        #X_train, y_train = X_train[0:1], y_train[0:1]
        #X_val, y_val = X_train, y_train

        # reshape data to N,W,H,C
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_val = X_val.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        # select data based on split
        if split == "train":
            X = X_train
            y = y_train
        elif split == "val":
            X = X_val
            y = y_val
        elif split == "test":
            X = X_test
            y = y_test

        # As a sanity check, we print out th size of the training and test data dimenstion
        #print ('Train data shape: ', X_train.shape)
        #print ('Train labels shape: ', y_train.shape)
        #print ('Validation data shape: ', X_val.shape)
        #print ('Validation labels shape: ', y_val.shape)
        #print ('Test data shape: ', X_test.shape)
        #print ('Test labels shape: ', y_test.shape)
        
        args.num_classes = len(classes)

        args.labels = {}
        args.idx2label = {}
        for i, l in enumerate(classes):
            args.labels[l] = i
            args.idx2label[i] = l
        args.label_names = classes
        
        self.data_indices = list(range(len(X)))
        random.shuffle(self.data_indices)

        self.args = args
        self.X = X
        self.y = y

        super().__init__(data_queue, self.data_indices, None, args.batch_size, repeat=repeat)

    def load_data(self, data_index):
        imgs, gts = self.X[data_index], self.y[data_index]
        meta = self.args.idx2label[gts]
        return imgs[np.newaxis, ...], gts[np.newaxis, ...], meta

    def collate(self, batch):
        imgs, gts, meta = list(zip(*batch))
        if len(imgs) > 0:
            imgs = np.concatenate(imgs, 0)
            gts = np.concatenate(gts, 0)
        return imgs, gts, meta

def test_process():
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'CIFAR10' #dataset name
    args.nworkers = 1
    args.batch_size = 2
    data_processes = []
    data_queue = Queue(8)
    for i in range(args.nworkers):
        data_processes.append(CIFAR10DataProcess(data_queue, args, split='train',
                                               repeat=False))
        data_processes[-1].start()
    N = len(data_processes[0].data_paths)
    batch_size = data_processes[0].batch_size
    Nb = int(N/batch_size)
    if Nb*batch_size < N:
        Nb += 1

    for imgs, gts, meta in get_while_running(data_processes, data_queue, 0):
        #check labels visually
        n, w, h, c = imgs.shape
        for i in range(len(imgs)):
            img = (imgs[i].reshape(w*h*c) + args.mean_image).reshape(w, h, c).astype(np.int32)
            show_image(img, gts[i], meta[i])
            #imgname = meta[i][1]
            #cv2.imshow(imgname,imgs[i])
            #cv2.waitKey(0)
            break
        break
    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()