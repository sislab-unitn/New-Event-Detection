# coding: utf-8

import imp
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
    num_tag = 0
    samples = []
    output = {}

    # triplet_pasted
    tp_true_positive = 0
    tp_false_positive = 0
    tp_true_negative = 0
    tp_false_negative = 0
    tp_num_tag = 0

    for idx, data in pbar:
        predicted = model(data["src"], data["src_mask"])
        attentions = predicted[-1]
        predicted = predicted[1].cpu()

        for pre_idx, predict_tags in enumerate(predicted):
            sentence_token_ids = data["sentence"][pre_idx][1:data["sentence_len"][pre_idx]-1]
            iob_tags = predict_tags[1:data["sentence_len"][pre_idx]-1, 0]
            word_ids = data["word_ids"][pre_idx][:data["sentence_len"][pre_idx]-2]
            original_iob_tags = data["original_iob_tag"][pre_idx][:data["original_iob_tag_len"][pre_idx]]
            assert sentence_token_ids.shape == iob_tags.shape
            predict_iob_aligned = []
            last_word_id = -1
            for iob_tag, word_id in zip(iob_tags, word_ids):
                if last_word_id == word_id:
                    continue
                if iob_tag == 1:
                    if len(predict_iob_aligned) == 0 or predict_iob_aligned[-1] == "o":
                        predict_iob_aligned.append("b")
                    else:
                        predict_iob_aligned.append("i")
                else:
                    predict_iob_aligned.append("o")
                last_word_id = word_id
            assert len(original_iob_tags) == len(predict_iob_aligned)

            for predict, gold in zip(predict_iob_aligned, original_iob_tags.tolist()):
                if predict == "i" or predict == "b":
                    if gold == 1:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if gold == 1:
                        false_negative += 1
                    else:
                        true_negative += 1
                num_tag += 1

            original_triplet_pasted_iob_tags = data["original_triplet_pasted_iob_tag"][pre_idx][:data["original_triplet_pasted_iob_tag_len"][pre_idx]]
            for predict, gold in zip(predict_iob_aligned, original_triplet_pasted_iob_tags.tolist()):
                if predict == "i" or predict == "b":
                    if gold == 1:
                        tp_true_positive += 1
                    else:
                        tp_false_positive += 1
                else:
                    if gold == 1:
                        tp_false_negative += 1
                    else:
                        tp_true_negative += 1
                tp_num_tag += 1

            samples.append({
                "narrative_key": dataset.id_to_narrative_key_dic[data["narrative_key"][pre_idx].item()],
                "sentence_key": dataset.id_to_narrative_key_dic[data["sentence_key"][pre_idx].item()],
                "sentence": data["sentence"][pre_idx].tolist(),
                "iob_tag": data["iob_tag"][pre_idx].tolist(),
                "predict_iob_tag": predict_iob_aligned,
                "previous_sentences": data["previous_sentences"][pre_idx].tolist(),
                "previous_tagged_words": data["previous_tagged_words"][pre_idx].tolist()
            })

    assert (true_positive+false_positive+true_negative+false_negative) == num_tag
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

    assert (tp_true_positive+tp_false_positive+tp_true_negative+tp_false_negative) == tp_num_tag
    if tp_true_positive+tp_false_positive > 0:
        tp_precision = tp_true_positive / (tp_true_positive+tp_false_positive)
    else:
        tp_precision = 0.0
    if tp_true_positive+tp_false_negative > 0:
        tp_recall = tp_true_positive / (tp_true_positive+tp_false_negative)
    else:
        tp_recall = 0.0
    if tp_precision + tp_recall > 0:
        tp_f1 = 2 * tp_precision * tp_recall / (tp_precision + tp_recall)
    else:
        tp_f1 = 0.0
    tp_precision *= 100
    tp_recall *= 100

    if return_results == True:
        return precision, recall, f1

    output["precision"] = precision
    output["recall"] = recall
    output["f1"] = f1
    output["samples"] = samples

    output["tp_precision"] = tp_precision
    output["tp_recall"] = tp_recall
    output["tp_f1"] = tp_f1

    loginfo_and_print(logger, "{:.2f},{:.2f},{:.4f},{:.2f},{:.2f},{:.4f}".format(precision, recall, f1, tp_precision, tp_recall, tp_f1))

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    jsonfilename = os.path.join(log_dir, hparams["model_name"]+".json")
    with open(jsonfilename, "w") as outfile:
        json.dump(output, outfile, ensure_ascii=False)
        outfile.write("\n")
