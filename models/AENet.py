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

from common import weights_init, FullModel


def AENet_setup(args):
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def AENet_create_model(args):
    """ Creates model """
    model = AENet(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion_rec = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    model = FullModel(args, model, criterion_rec, criterion_class)
    
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    
    model.apply(weights_init)
    return model


def AENet_step(args, item):
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
    inp = inp.transpose(3,2).transpose(2,1)
    loss, rec_err, outputs = args.model.forward(inp,targets)
    pred, decoded = outputs
    loss = loss.mean()

    class_acc = np.mean(torch.argmax(pred,1).detach().cpu().numpy()==gt) * 100

    losses = [loss, class_acc, rec_err]
    loss_names = ['loss','class_acc', 'rec_err']
    return loss_names, losses, outputs

class AENet_simple(nn.Module):
    def __init__(self, args):
        super(AENet, self).__init__()
        
        # Input size: [batch, 3, 32, 32]
        # Decoder Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8*8*4, 64),
            nn.ReLU(),
            nn.Linear(64, args.num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        x_encoded = encoded.reshape(x.shape[0], -1)
        pred = self.classifier(x_encoded)
        return pred, decoded

class AENet(nn.Module):
    def __init__(self, args):
        super(AENet, self).__init__()
        
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )

        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)   # [batch, 3, 32, 32]
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, args.num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        x_encoded = encoded.reshape(x.shape[0], -1)
        pred = self.classifier(x_encoded)
        return pred, decoded