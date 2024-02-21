# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewEventTagger(nn.Module):
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
            tgt = self.replace_pad(tgt)
            # tgt = tgt.cuda()
            loss = self.criterion(F.log_softmax(src, dim=1).transpose(2, 1)[:, :, :tgt.shape[1]], tgt.cuda())
            return loss
        else:
            topv, topi = F.log_softmax(src, dim=1).topk(self.hparams["num_class"])
            return topv, topi, attentions

    def replace_pad(self, tgt_list):
        for tgt in tgt_list:
            count = 0
            for idx in range(len(tgt)):
                if tgt[idx] == -100:
                    count += 1
                if count >= 2:
                    tgt[idx] = -100
        return tgt_list
