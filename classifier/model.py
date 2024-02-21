# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewEventDetectionClassifier(nn.Module):
    def __init__(self, hparams, bert, weight):
        super().__init__()
        self.training = True
        self.hparams = hparams
        self.bert = bert
        self.criterion = nn.NLLLoss(weight=torch.tensor(weight))

    def forward(self, src, src_mask, tgt=None):
        attentions = {}

        src = self.bert(src.cuda(), src_mask.cuda(), output_attentions=True)
        attentions["src"] = src.attentions[-1].mean(dim=1)[:, 0].tolist()
        src = src.logits

        if tgt is not None:
            # tgt = tgt.cuda()
            loss = self.criterion(F.log_softmax(src, dim=1), tgt[:, 0].cuda())
            return loss
        else:
            topv, topi = F.log_softmax(src, dim=1).topk(self.hparams["num_class"])
            return topv, topi, attentions
