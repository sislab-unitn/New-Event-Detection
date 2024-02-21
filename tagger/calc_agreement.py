# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import argparse
import json
from collections import defaultdict
from collections import Counter

import numpy as np

from utils.log_print import loginfo_and_print
from tagger.evaluation_script import compute_f1_coverage


def calc_agreements(iteration_samples, data):
    f1_list = []
    precision_list = []
    recall_list = []
    for iteration in iteration_samples:
        narrative_dic = {}
        for sample in iteration:
            narrative_key = sample["narrative_key"]
            if narrative_dic.get(narrative_key, None) is None:
                narrative_dic[narrative_key] = {}

            sentence_key = sample["sentence_key"]
            narrative_dic[narrative_key][sentence_key] = {
                "predict_iob": "".join(sample["predict_iob_tag"]),
                "gold_iob": "".join(data[narrative_key][sentence_key]["iob_tag"]),
                "sentence": data[narrative_key][sentence_key]["sentence"]
            }

        for narrative_data in narrative_dic.values():
            predict_iob_all = ""
            gold_iob_all = ""
            sentence_list = []
            for sentence_key in sorted(narrative_data.keys()):
                predict_iob_all += narrative_data[sentence_key]["predict_iob"]
                gold_iob_all += narrative_data[sentence_key]["gold_iob"]
                sentence_list.append(narrative_data[sentence_key]["sentence"])
            f1, precision, recall = compute_f1_coverage(predict_iob_all, gold_iob_all, " ".join(sentence_list), exact_match=True)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
    return np.array(f1_list).mean(), np.array(precision_list).mean(), np.array(recall_list).mean()


def format_data(data, pasted_only=False):
        data_dic = defaultdict(list)
        for batch in data:
            for narrative_key, narrative in batch.items():
                batch_name = narrative["annotations"]["batch"]
                if batch_name is None:
                    continue
                if "qualification" not in batch_name and "batch" not in batch_name:
                    continue
                data_dic[narrative_key].append(narrative)

        narrative_data = {}
        for narrative_key, annotation_list in data_dic.items():
            narrative_data[narrative_key] = {}
            for sentence_key in annotation_list[0].keys():
                if sentence_key == "annotations":
                    continue
                narrative_data[narrative_key][sentence_key] = {}
                narrative_data[narrative_key][sentence_key]["sentence"] = annotation_list[0][sentence_key]["sentence"]
                iob_list = []
                for annotation in annotation_list:
                    if pasted_only:
                        iob_list.append(annotation[sentence_key]["pasted_iob"])
                    else:
                        iob_list.append(annotation[sentence_key]["triplet_pasted_iob"])
                iob_tag = []
                for idx in range(len(iob_list[0])):
                    tag = max(Counter([iob[idx] for iob in iob_list]).items(), key=lambda x:x[1])[0]
                    if tag == "i" and iob_tag[-1] == "o":
                        tag = "b"
                    iob_tag.append(tag)
                narrative_data[narrative_key][sentence_key]["iob_tag"] = iob_tag
                tagged_words = []
                for word, tag in zip(narrative_data[narrative_key][sentence_key]["sentence"].split(" "), iob_tag):
                    if tag != "o":
                        tagged_words.append(word)
                narrative_data[narrative_key][sentence_key]["tagged_words"] = tagged_words

        return narrative_data


if __name__ == "__main__":
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


    path = "./data/send/annotation_result_iob/send.annotation.result.iob.20220728.json"
    with open(path, "r") as f:
        data = json.loads(f.read())
    data = format_data(data, pasted_only=True)

    # bert
    model = "bert-base-uncased"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.pasted".format(model, batch_size, learning_rate)

    bert_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(bert_logfilename, "r") as f:
        bert_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(bert_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "BERT (copy-pasted) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))

    # roberta
    model = "roberta-base"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.pasted".format(model, batch_size, learning_rate)

    roberta_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(roberta_logfilename, "r") as f:
        roberta_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(roberta_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "RoBERTa (copy-pasted) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))


    path = "./data/send/annotation_result_iob/send.annotation.result.iob.20220728.json"
    with open(path, "r") as f:
        data = json.loads(f.read())
    data = format_data(data, pasted_only=False)

    # bert
    model = "bert-base-uncased"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.pasted".format(model, batch_size, learning_rate)

    bert_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(bert_logfilename, "r") as f:
        bert_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(bert_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "BERT (copy-pasted tp test) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))

    # roberta
    model = "roberta-base"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.pasted".format(model, batch_size, learning_rate)

    roberta_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(roberta_logfilename, "r") as f:
        roberta_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(roberta_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "RoBERTa (copy-pasted tp test) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))


    # bert
    model = "bert-base-uncased"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.triplet_pasted".format(model, batch_size, learning_rate)

    bert_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(bert_logfilename, "r") as f:
        bert_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(bert_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "BERT (triplet_pasted) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))

    # roberta
    model = "roberta-base"
    batch_size = "8"
    learning_rate = "1e-5"

    log_prefix = "{}.{}.{}.triplet_pasted".format(model, batch_size, learning_rate)

    roberta_logfilename = os.path.join(log_dir, "integrate_results.{}.json".format(log_prefix))
    with open(roberta_logfilename, "r") as f:
        roberta_json = [json.loads(l.strip()) for l in f][0]

    f1, precision, recall = calc_agreements(roberta_json["samples"], data)
    # precision, recall, f1
    loginfo_and_print(logger, "RoBERTa (triplet_pasted) & ${:.2f}$ & ${:.2f}$ & ${:.4f}$ \\\\".format(f1, precision, recall))
