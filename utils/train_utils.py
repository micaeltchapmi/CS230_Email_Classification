# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time, sys
import logging
import random
import torchnet as tnt
import numpy as np
import torch
import torch.optim as optim
from multiprocessing import Queue
from tqdm import tqdm
from shared.dataprocess import kill_data_processes
from shared.datasets.cifar10 import CIFAR10DataProcess
from data_utils import save_prediction
from loss_utils import getLabelCount, getmeaniou, getconfmatrix
from data_process import show_image
import json
from AENet import *


def check_overwrite(fname):
    if os.path.isfile(fname):
        valid = ['y', 'yes', 'no', 'n']
        inp = None
        while inp not in valid:
            inp = input(
                '%s already exists. Do you want to overwrite it? (y/n)'
                % fname)
            if inp.lower() in ['n', 'no']:
                raise Exception('Please create new experiment.')


def data_setup(args, phase, num_workers, repeat):
    DataProcessClass = CIFAR10DataProcess
    # Initialize data processes
    data_queue = Queue(4 * num_workers)
    data_processes = []
    for i in range(num_workers):
        data_processes.append(DataProcessClass(data_queue, args, phase, repeat=repeat))
        data_processes[-1].start()
    return data_queue, data_processes


def parse_experiment(odir):
    #TODO: FIx this
    stats = json.loads(open(odir + '/trainlog.txt').read())
    valloss = [k['loss_val'] for k in stats if 'loss_val' in k.keys()]
    epochs = [k['epoch'] for k in stats if 'loss_val' in k.keys()]
    last_epoch = max(epochs)
    idx = np.argmin(valloss)
    best_val_loss = float('%.6f' % (valloss[idx]))
    best_epoch = epochs[idx]
    #val_results = odir + '/results_train_%d' % (best_epoch)
    #val_results = open(val_results).readlines()
    #first_line = val_results[0]
    #num_params = int(first_line.rstrip().split(' ')[-1])
    #fix num_params when cleaning
    num_params = None

    return last_epoch, best_epoch, best_val_loss, num_params


def model_at(args, i):
    return os.path.join(args.odir, 'models/model_%d.pth.tar' % (i))


def resume(args, i):
    """ Loads model and optimizer state from a previous checkpoint. """
    print("=> loading checkpoint '{}'".format(args.resume))
    if torch.cuda.is_available():
        device= "cuda"
    else:
        device='cpu'
    checkpoint = torch.load(args.resume, map_location=device)
    if 'args' not in list(checkpoint.keys()): # Pre-trained model?
        r_args = args
        model = eval(args.net + '_create_model')(r_args) #use original arguments, architecture can't change
        optimizer = create_optimizer(args, model)
        model.load_state_dict(checkpoint)
        checkpoint['epoch'] = 0
        args.start_epoch = None
    else:
        r_args = checkpoint['args']
        model = eval(args.net + '_create_model')(r_args) #use original arguments, architecture can't change
        args.nparams = r_args.nparams
        optimizer = create_optimizer(args, model)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']

    stats = json.loads(open(os.path.join(args.odir, 'trainlog.txt')).read())
    return model, optimizer, stats, r_args


def create_optimizer(args, model):
    if args.net == "SVM":
        return None

    params = filter(lambda p: p.requires_grad, model.parameters())
    print(args.optim)
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, betas=args.betas, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.wd)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, alpha=args.alpha, weight_decay=args.wd, momentum=args.momentum)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, rho=args.rho, weight_decay=args.wd)
    else:
        raise ValueError('Only Support SGD, adam, adagrad, rmsprop, and adadelta')
    return optimizer


def set_seed(seed, cuda=True):
    """ Sets seeds in all frameworks"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)


def train(args, epoch, data_queue, data_processes):
    """ Trains for one epoch """
    print("Training....")
    args.model.train()
    return train_or_test(args, data_queue, data_processes, training=True)


def test(split, args):
    """ Evaluated model on test set """
    print("Testing....")
    args.model.eval()

    data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
    with torch.no_grad():
        return train_or_test(args, data_queue, data_processes, training=False)


def train_or_test(args, data_queue, data_processes, training):
    """ Run on pass over dataset """
    N = len(data_processes[0].data_paths)
    batch_size = data_processes[0].batch_size
    Nb = int(N/batch_size)
    if Nb*batch_size < N:
        Nb += 1

    meters = []
    t0 = time.time()

    # iterate over dataset in batches
    for bidx in tqdm(range(Nb)):
        item = data_queue.get()
        imgs, gts, meta = item
        N, W, H, C = imgs.shape

        t_loader = 1000*(time.time()-t0)
        t0 = time.time()

        if training: args.optimizer.zero_grad()
        lnm, losses, outputs = args.step(args, item)
        if training: losses[0].backward()
        if training: args.optimizer.step()

        if len(meters) == 0:
            Nl = len(lnm)
            for i in range(Nl):
                meters.append(tnt.meter.AverageValueMeter())

        t_trainer = 1000*(time.time()-t0)
        for ix, l in enumerate(losses):
            meters[ix].add(l.item())

        if (bidx % 50) == 0:
            prt = 'Train '
            for ix in range(Nl):
                prt += '%s %f, ' % (lnm[ix], losses[ix].item())
            prt += 'Loader %f ms, Train %f ms. \n' % (t_loader, t_trainer)
            print(prt)
        log = 'Batch '
        for ilx in range(Nl):
            log += '%s %f, ' % (lnm[ilx], losses[ilx].item())
        log += 'Loader time %f ms, Trainer time %f ms.' % (t_loader, t_trainer)
        logging.debug(log)
        t0 = time.time()

    return lnm, [meters[ix].value()[0] for ix in range(Nl)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def metrics(split, args, epoch=0):
    with torch.no_grad():
        print("Metrics ....")
        args.model.eval()
        data_queue, data_processes = data_setup(args, split, num_workers=1, repeat=False)
        N = len(data_processes[0].data_paths)
        batch_size = data_processes[0].batch_size
        Nb = int(N/batch_size)
        overall_class_acc = []
        preds = []
        truths = []
        recs = []
        if Nb*batch_size < N:
            Nb += 1
        # iterate over dataset in batches
        count = 0

        for bidx in tqdm(range(Nb)):
            item = data_queue.get()
            imgs, gts, meta = item
            N, W, H, C = imgs.shape
            lnm, losses, outputs = args.step(args, item)

            overall_class_acc.append(losses[1])
            pred_i = outputs[0].cpu().numpy()
            rec_i = outputs[1].cpu().numpy()

            preds.extend(pred_i)
            truths.extend(gts)
            recs.extend(rec_i)
            
            #save one image from each batch, and at most 5 images for viewing sample predictions
            if count < 5:
                view_predictions(args, imgs, gts, pred_i, rec_i, meta, bidx, epoch)
                count+=1

        preds = np.asarray(preds)
        truths = np.asarray(truths)
        
        overall_class_acc = np.mean(overall_class_acc)

        odir = args.odir + '/acc'
        os.makedirs(odir, exist_ok=True)
        outfile = odir + '/results_%s_%d.txt' % (split, epoch + 1)
        
        #save average keypoint distance
        print("Saving results to %s ..." % (outfile))
        with open(outfile, 'w') as f:
            f.write('classification acc: %.5f\n' % (overall_class_acc))


def view_predictions(args, imgs, gts, preds, recs, meta, bid, epoch):
    count = 0
    max_save = 1 # number of images to save per batch: originally 1
    print("Plotting predictions")
    for j in range(len(preds)):
        rec = recs[j]
        pred = np.argmax(preds[j])
        gt = gts[j]
        true_label = args.idx2label[gt]
        pred_label = args.idx2label[pred]
        imgname = "epoch%d_T_%s_P_%s" % (epoch, true_label, pred_label)
        img = imgs[j]
        save_prediction(args, img, rec, imgname, bid, j, epoch)
        count+=1
        if count >= max_save:
            break
    return
