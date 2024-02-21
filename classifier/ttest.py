# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import argparse
import json
from collections import defaultdict
from collections import Counter

from scipy.stats import ttest_rel

import torch

from utils.log_print import loginfo_and_print


def get_argparse():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-l", "--log_dir",
    #     help="log file directory",
    #     required=True
    # )

    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, os.path.splitext(os.path.basename(__file__))[0]+".log")
    format = "%(message)s"
    logger = logging.getLogger("logger")
    handler = logging.FileHandler(filename=logfilename, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # bert
    model = "bert-base-uncased"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}".format(model, batch_size, learning_rate)

    bert_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(bert_logfilename, "r") as f:
        bert_json = [json.loads(l.strip()) for l in f][0]

    bert_data = torch.tensor([bert_json["precision"], bert_json["recall"], bert_json["f1"]])
    bert_data = bert_data.view(3, -1)
    bert_mean = bert_data.mean(dim=1)
    bert_std = bert_data.std(dim=1)

    # roberta
    model = "roberta-base"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}".format(model, batch_size, learning_rate)

    roberta_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(roberta_logfilename, "r") as f:
        roberta_json = [json.loads(l.strip()) for l in f][0]

    roberta_data = torch.tensor([roberta_json["precision"], roberta_json["recall"], roberta_json["f1"]])
    roberta_data = roberta_data.view(3, -1)
    roberta_mean = roberta_data.mean(dim=1)
    roberta_std = roberta_data.std(dim=1)

    # accuracy, r@5, mrr
    # loginfo_and_print(logger, "BERT & ${:.2f}\\ (\\pm{:.2f})$ & ${:.2f}\\ (\\pm{:.2f})$ & ${:.4f}\\ (\\pm{:.4f})$ \\\\".format(bert_mean[0], bert_std[0], bert_mean[1], bert_std[1], bert_mean[2], bert_std[2]))
    # loginfo_and_print(logger, "RoBERTa & ${:.2f}\\ (\\pm{:.2f})$ & ${:.2f}\\ (\\pm{:.2f})$ & ${:.4f}\\ (\\pm{:.4f})$ \\\\".format(roberta_mean[0], roberta_std[0], roberta_mean[1], roberta_std[1], roberta_mean[2], roberta_std[2]))
    loginfo_and_print(logger, "\\textbf{{BERT}} & ${:.1f}$ & ${:.1f}$ & ${:.1f}$ \\\\".format(bert_mean[0], bert_mean[1], bert_mean[2]*100))
    loginfo_and_print(logger, "\\textbf{{RoBERTa}} & ${:.1f}$ & ${:.1f}$ & ${:.1f}$ \\\\".format(roberta_mean[0], roberta_mean[1], roberta_mean[2]*100))
