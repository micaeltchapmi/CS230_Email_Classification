# ------------------------------------------------------------------------------
#Copyright 2021, Micael Tchapmi, All rights reserved
# ------------------------------------------------------------------------------

from transformers import BertTokenizer

import torch
import torch.nn as nn
import torch._utils
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from transformers import BertModel
import pandas as pd

from common import weights_init, FullModel

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def BERT_setup(args):
    args.odir = 'results/%s/%s' % (args.dataset, args.net)
    args.odir += '_b%d' % args.batch_size


def BERT_create_model(args):
    """ Creates model """
    model = BERT(args)
    args.gpus = list(range(torch.cuda.device_count()))
    args.nparams = sum([p.numel() for p in model.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    criterion_class = nn.CrossEntropyLoss()

    
    #model = FullModel(args, model, criterion_class)
    
    if len(args.gpus) > 0:
        model = nn.DataParallel(model, device_ids=args.gpus).cuda()
    
    
    return model


def BERT_step(args, item):
    text_vectors, gt, meta = item
    n, d = text_vectors.shape
    
    df_text =pd.DataFrame(meta)
    texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                       return_tensors="pt") for text in df_text["Text"]]
    
    
    input_ids = []
    masks = []
    for i in range(len(texts)):

        input_id = texts[i]["input_ids"]
        input_ids.append(input_id)
        mask = texts[i]["attention_mask"]
        masks.append(mask)
    input_ids = torch.stack(input_ids)

    input_ids = torch.squeeze(input_ids,dim=1)

    masks = torch.stack(masks)
    masks = torch.squeeze(masks,dim=1)  
    


    args.gpus = list(range(torch.cuda.device_count()))
    if len(args.gpus) > 0:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long().cuda()
        
    else:
        targets = Variable(torch.from_numpy(gt), requires_grad=False).long()
        

    targets = targets.contiguous()

    pred = args.model.forward(input_ids, masks)
    
    classification_loss = nn.CrossEntropyLoss()(pred, targets)
    loss = classification_loss    
    loss = loss.mean()

    class_acc = np.mean(torch.argmax(pred,1).detach().cpu().numpy()==gt) * 100

    losses = [loss, class_acc]
    loss_names = ['loss', 'class_acc']
    return loss_names, losses, pred

class BERT(nn.Module):
    def __init__(self, args, dropout =0.5):
        super(BERT, self).__init__()
        
        # Input size: [batch, 3, 100, 100]
        self.args = args
        self.bert = BertModel.from_pretrained('bert-base-cased')
        # Freeze all the parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, 2)


    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        final_layer = self.linear(pooled_output)
        
        return final_layer