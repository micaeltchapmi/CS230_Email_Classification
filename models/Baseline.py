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


def Baseline_setup(args):
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def Baseline_create_model(args):
    """ Creates model """
    model = Baseline(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion_class = nn.CrossEntropyLoss()
    model = FullModel(args, model, criterion_class)
    
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    
    model.apply(weights_init)
    return model


def Baseline_step(args, item):
    text_vectors, gt, meta = item
    n, d = text_vectors.shape
    args.gpus = list(range(torch.cuda.device_count()))
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long().cuda()
        inp = Variable(torch.from_numpy(text_vectors), requires_grad=False).float().cuda()
    else:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long()
        inp = Variable(torch.from_numpy(text_vectors), requires_grad=False).float()

    targets = targets.contiguous()
    loss, pred = args.model.forward(inp,targets)
    loss = loss.mean()

    class_acc = np.mean(torch.argmax(pred,1).detach().cpu().numpy()==gt) * 100

    losses = [loss, class_acc]
    loss_names = ['loss', 'class_acc']
    return loss_names, losses, pred

class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()
        
        # Input size: [batch, 3, 100, 100]
        self.args = args

        self.linear = nn.Sequential(
            nn.Linear(1000, self.args.num_classes) #sizeof_word_vector -> num_classes
        )

    def forward(self, x):
        pred = self.linear(x)
        return pred