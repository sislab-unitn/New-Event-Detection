# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import tqdm
import pickle
from collections import Counter
import json
import random

import numpy as np

from utils.log_print import loginfo_and_print
from tagger.evaluation_script import compute_f1_coverage


logger = logging.getLogger("logger").getChild("test")

random.seed(0)


def baseline_test(dataset, method, pasted_only=False):
    dataset.switch_data("test")
    true_positive_list = []
    false_positive_list = []
    true_negative_list = []
    false_negative_list = []
    num_tag_list = []
    output = {}
    agg_f1_list = []
    agg_precision_list = []
    agg_recall_list = []

    for data_idx, data in enumerate(dataset.data):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        num_tag = 0

        predict_tag_all = ""
        gold_tag_all = ""
        sentence_list = []
        for sentence_idx, (sentence, iob_tag) in enumerate(zip(data["sentences"], data["iob_tag"])):
            predict_tag = []
            tag_len = len(iob_tag)
            for idx in range(tag_len):
                if method == "random":
                    if random.randint(0, 1) == 1:
                        predict_tag.append("i")
                    else:
                        predict_tag.append("o")
                elif method == "beginning":
                    if idx <= int(tag_len*0.3):
                        predict_tag.append("i")
                    else:
                        predict_tag.append("o")
                elif method == "end":
                    if idx >= int(tag_len*0.7):
                        predict_tag.append("i")
                    else:
                        predict_tag.append("o")

            assert len(predict_tag) == len(iob_tag)
            for pred, tgt in zip(predict_tag, iob_tag):
                if pred == "i":
                    if tgt == "i":
                        true_positive += 1
                        # print(selected_idx, True, len(triplet_label_list), triplet_label_list)
                    else:
                        false_positive += 1
                        # print(selected_idx, False, len(triplet_label_list), triplet_label_list)
                else:
                    if tgt == "i":
                        false_negative += 1
                    else:
                        true_negative += 1

                num_tag += 1

            for idx in range(len(predict_tag)):
                if idx == 0 and predict_tag[idx] == "i":
                    predict_tag[idx] = "b"
                if idx > 0 and predict_tag[idx-1] == "o" and predict_tag[idx] == "i":
                    predict_tag[idx] = "b"

                if idx == 0 and iob_tag[idx] == "i":
                    iob_tag[idx] = "b"
                elif idx > 0 and iob_tag[idx-1] == "o" and iob_tag[idx] == "i":
                    iob_tag[idx] = "b"

            predict_tag_all += "".join(predict_tag)
            gold_tag_all += "".join(iob_tag)
            sentence_list.append(sentence)
        agg_f1, agg_precision, agg_recall = compute_f1_coverage(predict_tag_all, gold_tag_all, " ".join(sentence_list), exact_match=True)
        agg_f1_list.append(agg_f1)
        agg_precision_list.append(agg_precision)
        agg_recall_list.append(agg_recall)

        true_positive_list.append(true_positive)
        false_positive_list.append(false_positive)
        true_negative_list.append(true_negative)
        false_negative_list.append(false_negative)
        num_tag_list.append(num_tag)

    true_positive = sum(true_positive_list)
    false_positive = sum(false_positive_list)
    true_negative = sum(true_negative_list)
    false_negative = sum(false_negative_list)
    num_tag = sum(num_tag_list)

    assert (true_positive+false_positive+true_negative+false_negative) == num_tag
    precision = true_positive / (true_positive+false_positive)
    recall = true_positive / (true_positive+false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    precision *= 100
    recall *= 100

    agg_f1 = np.array(agg_f1_list).mean()
    agg_precision = np.array(agg_precision_list).mean()
    agg_recall = np.array(agg_recall_list).mean()

    output["precision"] = precision
    output["recall"] = recall
    output["f1"] = f1
    output["agg_f1"] = agg_f1
    output["agg_precision"] = agg_precision
    output["agg_recall"] = agg_recall

    loginfo_and_print(logger, "{:.2f},{:.2f},{:.4f},{:.2f},{:.2f},{:.2f}".format(precision, recall, f1, agg_f1, agg_precision, agg_recall))

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    if pasted_only:
        jsonfilename = os.path.join(log_dir, method+".pasted.json")
    else:
        jsonfilename = os.path.join(log_dir, method+".triplet_pasted.json")
    with open(jsonfilename, "w") as outfile:
        json.dump(output, outfile, ensure_ascii=False)
        outfile.write("\n")
