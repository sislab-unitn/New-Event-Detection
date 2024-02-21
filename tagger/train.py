# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader

from utils.log_print import loginfo_and_print
from utils.batch_sampler import collate_fn
from utils.batch_sampler import RandomBatchSampler

from tagger.test import test


logger = logging.getLogger("logger").getChild("train")


def run_epochs(hparams, model, dataset, model_path):
    learning_rate = hparams["learning_rate"]
    decay_step = hparams["decay_step"]
    lr_decay = hparams["lr_decay"]
    max_epoch = hparams["max_epoch"]
    batch_size = hparams["batch_size"]
    max_gradient = hparams["max_gradient"]

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=lr_decay)
    # scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=lr_decay, total_iters=decay_step)
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/20, total_iters=20)
    # decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=lr_decay)

    valid_loss, valid_precision, valid_recall, valid_f1 = valid(model, dataset, batch_size, hparams)
    loginfo_and_print(
        logger,
        "Valid (Epoch {}): {:.4f}, {:.2f}, {:.2f}, {:.4f}".format(0, valid_loss, valid_precision, valid_recall, valid_f1)
    )
    last_valid_loss = 10000000
    last_valid_accuracy = -1
    last_valid_recall = -1
    last_valid_f1 = -1
    for epoch in range(1, max_epoch+1):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn,
            drop_last=True, num_workers=2,
            sampler=RandomBatchSampler(dataset, batch_size)
        )
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        loss = 0
        for idx, data in pbar:
            loss += iteration(model, data, hparams, optimizer, max_gradient)
            scheduler.step()
            # decay_scheduler.step()
            pbar.set_description("Train (Epoch {}): {:.4f}".format(epoch, loss/(idx+1)))

        logger.info("Train (Epoch {}): {:.4f}".format(epoch, loss/(idx+1)))
        valid_loss, valid_precision, valid_recall, valid_f1 = valid(model, dataset, batch_size, hparams)
        loginfo_and_print(
            logger,
            "Valid (Epoch {}): {:.4f}, {:.2f}, {:.2f}, {:.4f}".format(epoch, valid_loss, valid_precision, valid_recall, valid_f1)
        )
        if valid_loss > last_valid_loss and valid_precision < last_valid_precision:
            loginfo_and_print(
                logger,
                "Valid performances decreased compared to the last values. Train terminated."
            )
            break
        last_valid_loss = valid_loss
        last_valid_precision = valid_precision
        last_valid_recall = valid_recall
        last_valid_f1 = valid_f1

        torch.save({
            "epoch": epoch,
            "hparams": hparams,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict(),
        }, model_path)

        # dataset.resampling_data()
    return last_valid_precision, last_valid_recall


def valid(model, dataset, batch_size, hparams):
    dataset.switch_data("valid")
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        drop_last=True, num_workers=0,
        sampler=RandomBatchSampler(dataset, batch_size)
    )
    loss = 0
    for idx, data in enumerate(dataloader):
        loss += iteration(model, data, hparams)
    loss /= (idx+1)

    precision, recall, f1 = test(hparams, model, dataset, return_results=True)

    dataset.switch_data("train")
    model.train()

    return loss, precision, recall, f1


def iteration(model, data, hparams, optimizer=None, max_gradient=None):
    with torch.set_grad_enabled(model.training):
        if model.training:
            optimizer.zero_grad()

        loss = model(
            data["src"], data["src_mask"],
            data["iob_tag"]
        )

        if model.training:
            loss.backward()
            # _ = clip_grad_norm(model.parameters(), max_gradient)
            optimizer.step()

    return loss.item()
