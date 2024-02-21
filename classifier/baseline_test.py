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

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger").getChild("test")

random.seed(0)


def baseline_test(dataset, method):
    dataset.switch_data("test")
    true_positive_list = []
    false_positive_list = []
    true_negative_list = []
    false_negative_list = []
    num_triplet_list = []
    output = {}

    for idx, data in enumerate(dataset.data):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        num_triplet = 0
        known_entities = []

        for sentence_idx, (triplet_label_list, triplet_position_list) in enumerate(zip(data["triplet_labels"], data["triplet_positions"])):
            if len(triplet_label_list) <= 0:
                continue

            selected_idx_list = []
            if method == "random_one":
                selected_idx_list.append(random.randint(0, len(triplet_label_list)-1))
            elif method == "random_each":
                for idx in range(len(triplet_label_list)):
                    if random.randint(0, 1) == 1:
                        selected_idx_list.append(idx)
            elif method == "first":
                selected_idx_list.append(triplet_position_list.index(
                    min(triplet_position_list)
                ))
            elif method == "last":
                selected_idx_list.append(triplet_position_list.index(
                    max(triplet_position_list)
                ))
            elif method == "subject":
                for triplet_idx in range(len(triplet_label_list)):
                    subject = data["subjects"][sentence_idx][triplet_idx]
                    if subject not in known_entities:
                        selected_idx_list.append(triplet_idx)
                        known_entities.append(subject)
                    # subject_coref_id = data["subject_coref_id"][sentence_idx][triplet_idx]
                    # for coref_id in subject_coref_id:
                    #     if coref_id not in known_entities:
                    #         selected_idx_list.append(triplet_idx)
                    #         known_entities.extend(subject_coref_id)
                    #         known_entities = list(set(known_entities))
                    #         break
            elif method == "entity":
                for triplet_idx in range(len(triplet_label_list)):
                    subject = data["subjects"][sentence_idx][triplet_idx]
                    object_ = data["objects"][sentence_idx][triplet_idx]
                    if subject not in known_entities or object_ not in known_entities:
                        selected_idx_list.append(triplet_idx)
                        known_entities.append(subject)
                        known_entities.append(object_)
                        known_entities = list(set(known_entities))
                    # subject_coref_id = data["subject_coref_id"][sentence_idx][triplet_idx]
                    # object_coref_id = data["object_coref_id"][sentence_idx][triplet_idx]
                    # for coref_id in subject_coref_id+object_coref_id:
                    #     if coref_id not in known_entities:
                    #         selected_idx_list.append(triplet_idx)
                    #         known_entities.extend(subject_coref_id+object_coref_id)
                    #         known_entities = list(set(known_entities))
                    #         break

            for selected_idx in selected_idx_list:
                if triplet_label_list[selected_idx] == 1:
                    true_positive += 1
                    # print(selected_idx, True, len(triplet_label_list), triplet_label_list)
                else:
                    false_positive += 1
                    # print(selected_idx, False, len(triplet_label_list), triplet_label_list)
            for idx in range(len(triplet_label_list)):
                if idx in selected_idx_list:
                    continue
                if triplet_label_list[idx] == 1:
                    false_negative += 1
                else:
                    true_negative += 1

            num_triplet += len(triplet_label_list)

        true_positive_list.append(true_positive)
        false_positive_list.append(false_positive)
        true_negative_list.append(true_negative)
        false_negative_list.append(false_negative)
        num_triplet_list.append(num_triplet)

    true_positive = sum(true_positive_list)
    false_positive = sum(false_positive_list)
    true_negative = sum(true_negative_list)
    false_negative = sum(false_negative_list)
    num_triplet = sum(num_triplet_list)

    assert (true_positive+false_positive+true_negative+false_negative) == num_triplet
    precision = true_positive / (true_positive+false_positive)
    recall = true_positive / (true_positive+false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    precision *= 100
    recall *= 100

    output["precision"] = precision
    output["recall"] = recall
    output["f1"] = f1

    loginfo_and_print(logger, "{:.2f},{:.2f},{:.4f}".format(precision, recall, f1))

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    jsonfilename = os.path.join(log_dir, method+".json")
    with open(jsonfilename, "w") as outfile:
        json.dump(output, outfile, ensure_ascii=False)
        outfile.write("\n")
