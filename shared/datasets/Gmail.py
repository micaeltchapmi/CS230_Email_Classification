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
from data_utils import show_image, get_CIFAR10_data, preprocessing_CIFAR10_data, load_CIFAR10, visualize_sample
sys.path.insert(0, './')


class Gmail_DataProcess(DataProcess):

    def __init__(self, data_queue, args, split='train', repeat=True):
        """CDI dataloader.
        Args:
            data_queue: multiprocessing queue where data is stored at.
            split: str in ('train', 'val', 'test'). Loads corresponding dataset.
            repeat: repeats epoch if true. Terminates after one epoch otherwise.
        """

        # Load dataset
        args.DATA_PATH = "./processed_data/%s" % (split)
        labels = ["Keep", "Delete"]
        args.labelsdict = {}
        args.idx2label = {}

        for i, l in enumerate(labels):
            args.labelsdict[l] = i
            args.idx2label[i] = str(l)
        args.num_classes = len(args.labelsdict)

        data_paths = [f for f in glob.glob(args.DATA_PATH+"/*/*/*")]

        self.data_paths = data_paths
        self.args = args
        random.shuffle(self.data_paths)
        super().__init__(data_queue, self.data_paths, None, args.batch_size, repeat=repeat)

    def getSample(self, args, fname):
        #read img local computer
        email_dict = json.load(open(fname))
        text_vector = np.asarray(email_dict["Text_Vector"])
        gt = np.asarray(email_dict["Label"]).astype(np.int32)
        meta = email_dict

        return text_vector, gt, meta
    
    def load_data(self, fname):
        text_vectors, gts, meta = self.getSample(self.args, fname)
        return text_vectors[np.newaxis, ...], gts[np.newaxis, ...], meta

    def collate(self, batch):
        text_vectors, gts, meta = list(zip(*batch))
        if len(text_vectors) > 0:
            text_vectors = np.concatenate(text_vectors, 0)
            gts = np.concatenate(gts, 0)
        return text_vectors, gts, meta

def test_process():
    from multiprocessing import Queue
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.dataset = 'Gmail' #dataset name
    args.nworkers = 1
    args.batch_size = 2
    data_processes = []
    data_queue = Queue(8)
    for i in range(args.nworkers):
        data_processes.append(Gmail_DataProcess(data_queue, args, split='train',
                                               repeat=False))
        data_processes[-1].start()
    N = len(data_processes[0].data_paths)
    batch_size = data_processes[0].batch_size
    Nb = int(N/batch_size)
    if Nb*batch_size < N:
        Nb += 1

    for text_vectors, gts, meta in get_while_running(data_processes, data_queue, 0):
        #check data is loaded properly with correct shape
        print(text_vectors.shape)
        print(gts.shape)
        print(len(meta))
        break
    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()