# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import tqdm
import pickle
from collections import Counter
import json

import torch
from torch.utils.data import DataLoader

from utils.log_print import loginfo_and_print
from utils.batch_sampler import collate_fn
from utils.batch_sampler import RandomBatchSampler

from classifier.dataset import Dataset


logger = logging.getLogger("logger").getChild("test")


def test(hparams, model, dataset, return_results=False):
    batch_size = hparams["batch_size"]
    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        drop_last=False, num_workers=0,
        sampler=RandomBatchSampler(dataset, batch_size)
    )
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    num_triplet = 0
    true_positive_samples = []
    false_positive_samples = []
    true_negative_samples = []
    false_negative_samples = []
    output = {}

    for idx, data in pbar:
        predicted = model(data["src"], data["src_mask"])
        attentions = predicted[-1]
        predicted = predicted[1].cpu()

        for pre_idx, predict in enumerate(predicted):
            if predict[0] == 1:
                if data["triplet_label"][pre_idx] == 1:
                    true_positive += 1
                    true_positive_samples.append({
                        "triplet_key": data["triplet_key"][pre_idx].tolist(),
                        "sentence": data["sentence"][pre_idx].tolist(),
                        "triplet": data["triplet"][pre_idx].tolist(),
                        "previous_sentences": data["previous_sentences"][pre_idx].tolist(),
                        "previous_events": data["previous_events"][pre_idx].tolist()
                    })
                else:
                    false_positive += 1
                    false_positive_samples.append({
                        "triplet_key": data["triplet_key"][pre_idx].tolist(),
                        "sentence": data["sentence"][pre_idx].tolist(),
                        "triplet": data["triplet"][pre_idx].tolist(),
                        "previous_sentences": data["previous_sentences"][pre_idx].tolist(),
                        "previous_events": data["previous_events"][pre_idx].tolist()
                    })
            else:
                if data["triplet_label"][pre_idx] == 1:
                    false_negative += 1
                    false_negative_samples.append({
                        "triplet_key": data["triplet_key"][pre_idx].tolist(),
                        "sentence": data["sentence"][pre_idx].tolist(),
                        "triplet": data["triplet"][pre_idx].tolist(),
                        "previous_sentences": data["previous_sentences"][pre_idx].tolist(),
                        "previous_events": data["previous_events"][pre_idx].tolist()
                    })
                else:
                    true_negative += 1
                    true_negative_samples.append({
                        "triplet_key": data["triplet_key"][pre_idx].tolist(),
                        "sentence": data["sentence"][pre_idx].tolist(),
                        "triplet": data["triplet"][pre_idx].tolist(),
                        "previous_sentences": data["previous_sentences"][pre_idx].tolist(),
                        "previous_events": data["previous_events"][pre_idx].tolist()
                    })
            num_triplet += 1

    assert (true_positive+false_positive+true_negative+false_negative) == num_triplet
    if true_positive+false_positive > 0:
        precision = true_positive / (true_positive+false_positive)
    else:
        precision = 0.0
    if true_positive+false_negative > 0:
        recall = true_positive / (true_positive+false_negative)
    else:
        recall = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    precision *= 100
    recall *= 100

    if return_results == True:
        return precision, recall, f1

    output["precision"] = precision
    output["recall"] = recall
    output["f1"] = f1
    output["true_positive_samples"] = true_positive_samples
    output["false_positive_samples"] = false_positive_samples
    output["true_negative_samples"] = true_negative_samples
    output["false_negative_samples"] = false_negative_samples

    loginfo_and_print(logger, "{:.2f},{:.2f},{:.4f}".format(precision, recall, f1))

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    jsonfilename = os.path.join(log_dir, hparams["model_name"]+".json")
    with open(jsonfilename, "w") as outfile:
        json.dump(output, outfile, ensure_ascii=False)
        outfile.write("\n")
