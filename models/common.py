from cProfile import label
import torch.nn as nn
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    def __init__(self, args, model, loss_rec, loss_class):
        super(FullModel, self).__init__()
        self.model = model
        self.loss_rec = loss_rec
        self.loss_class = loss_class
        self.args = args

    def forward(self, inputs, labels):
        outputs = self.model(inputs)
        pred, decoded = outputs

        rec_loss = self.loss_rec(decoded, inputs)
        classification_loss = self.loss_class(pred, labels)
        loss = rec_loss + classification_loss
        return torch.unsqueeze(loss,0), rec_loss, outputs
