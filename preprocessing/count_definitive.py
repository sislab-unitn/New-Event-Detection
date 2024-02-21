# coding: utf-8

# import random
import os
from random import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tqdm
import pickle
import logging
import collections
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
import json
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger").getChild("dataset")

np.random.seed(0)


def count_definitive(data_path):
    print("Reading files...")
    with open(data_path, "r") as f:
        data = json.loads(f.read())

    data_dic = defaultdict(list)
    for batch in data:
        for narrative_key, narrative in batch.items():
            batch_name = narrative["annotations"]["batch"]
            if batch_name is None:
                continue
            if "batch" not in batch_name:
                continue
            data_dic[narrative_key].append(narrative)

    def count_definitive(text):
        count = 0
        text = text.lower().split(" ")
        definitive_words = ["the"]
        for definitive in definitive_words:
            for word in text:
                if definitive == word:
                    count += 1
        return count

    def count_nondefinitive(text):
        count = 0
        text = text.lower().split(" ")
        nondefinitive_words = ["a", "an"]
        for nondefinitive in nondefinitive_words:
            for word in text:
                if nondefinitive == word:
                    count += 1
        return count

    num_definitive_triplet = 0
    num_nondefinitive_triplet = 0
    num_triplet = 0
    num_definitive_pasted = 0
    num_nondefinitive_pasted = 0
    num_pasted = 0
    for narrative_key, annotation_list in data_dic.items():
        for sentence_key in annotation_list[0].keys():
            if sentence_key == "annotations":
                continue
            for annotation in annotation_list:
                triplets = annotation[sentence_key]["triplets"]
                correct_triplets = annotation[sentence_key].get(
                    "correct_triplets", []
                )
                for triplet_key in correct_triplets:
                    triplet_text = re.sub(
                        "( ->| -|\[|\])+", "", triplets[triplet_key]["triplet"]
                    )
                    num_definitive_triplet += count_definitive(triplet_text)
                    num_nondefinitive_triplet += count_nondefinitive(triplet_text)
                    num_triplet += 1
                new_triplets = annotation[sentence_key].get(
                    "new_triplets", []
                )
                for pasted in new_triplets:
                    num_definitive_pasted += count_definitive(pasted)
                    num_nondefinitive_pasted += count_nondefinitive(pasted)
                    num_pasted += 1

    loginfo_and_print(logger, "Average number of definitive words included in triplet candidates: {}".format(num_definitive_triplet/num_triplet))
    loginfo_and_print(logger, "Average number of nondefinitive words included in triplet candidates: {}".format(num_nondefinitive_triplet/num_triplet))
    loginfo_and_print(logger, "Average number of definitive words included in copy-pasted events: {}".format(num_definitive_pasted/num_pasted))
    loginfo_and_print(logger, "Average number of nondefinitive words included in copy-pasted events: {}".format(num_nondefinitive_pasted/num_pasted))

    return


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "cout_definitive.log")
    format = "%(message)s"
    logger = logging.getLogger("logger")
    handler = logging.FileHandler(filename=logfilename, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    data_path = "./data/send/annotation_result_iob/send.annotation.result.iob.20220728.json"
    count_definitive(data_path)
