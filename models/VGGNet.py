# ------------------------------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch._utils
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torchvision.models as models
import os
import json
import pandas as pd

from common import weights_init, FullModel


def VGGNet_setup(args):
    #setup path to save experiment results
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def VGGNet_create_model(args):
    """ Creates model """
    model = VGGNet(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion_class = nn.CrossEntropyLoss()
    model = FullModel(args, model, criterion_class)
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        model = nn.DataParallel(model, device_ids=args.gpus)
    #model.apply(weights_init)
    return model


def VGGNet_step(args, item):
    imgs, gt, meta = item
    n,w,h,c = imgs.shape
    args.gpus = list(range(torch.cuda.device_count()))
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long().cuda()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float().cuda()
    else:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long()
        inp = Variable(torch.from_numpy(imgs), requires_grad=False).float()

    targets = targets.contiguous()
    #inp = inp.transpose(3,2).transpose(2,1)/255.0
    # normalize input image to range [-1, 1]
    inp = ( (inp - 127.5) / 127.5 ).transpose(3,2).transpose(2,1)
    loss, pred = args.model.forward(inp,targets)
    loss = loss.mean()

    class_acc = np.mean(torch.argmax(pred,1).detach().cpu().numpy()==gt) * 100

    losses = [loss, class_acc]
    loss_names = ['loss', 'class_acc']
    return loss_names, losses, pred


class VGGNet(nn.Module):
    def __init__(self, args):
        super(VGGNet, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        vgg16.classifier[6] = nn.Linear(4096, args.num_classes)
        self.encoder = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = self.avgpool(encoded)
        x_encoded = pooled.reshape(x.shape[0], -1)
        pred = self.classifier(x_encoded)
        return pred
