# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import tqdm
import json
import csv
from collections import Counter
import numpy as np
import spacy

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger")
os.makedirs("./preprocessing/log", exist_ok=True)


def read_rocstories(csv_path, json_path):
    with open(csv_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [r for r in csv_reader]
        header = data[0]
        data = data[1:]

    json_data = []
    with open(json_path, "w") as f:
        for d in data:
            if "train" in json_path:
                sentence = " ".join(d[2:])
            elif "valid_right" in json_path:
                sentence = d[4+int(d[7])]
            elif "valid_wrong" in json_path:
                sentence = d[4+(2 if int(d[7]) == 1 else 1)]
            elif "valid" in json_path:
                sentence = " ".join(d[1:5])
            else:
                sentence = " ".join(d[1:5])
            json_data.append({
                "id": d[0],
                "sentence": sentence
            })
            json.dump(json_data[-1], f, ensure_ascii=False)
            f.write("\n")

    return json_data


def analyze_corpus(corpus, logger):
    nlp = spacy.load("en_core_web_lg")
    doc_len = []
    uttr_len = []
    word_list = []
    pbar = tqdm.tqdm(corpus, total=len(corpus))
    for d in pbar:
        document = nlp(d["sentence"])
        doc_len.append(len([s for s in document.sents]))
        uttr_len.extend([len([t for t in s]) for s in document.sents])
        word_list.extend([t.text.lower() for s in document.sents for t in s])
    word_count = Counter(word_list)
    doc_len = np.array(doc_len)
    uttr_len = np.array(uttr_len)

    loginfo_and_print(logger, "Corpus size: {}".format(len(corpus)))
    loginfo_and_print(logger, "Ave. narrative length: {:.2f} (+-{:.4f})".format(doc_len.mean(), doc_len.std()))
    loginfo_and_print(logger, "Ave. utterance length: {:.2f} (+-{:.4f})".format(uttr_len.mean(), uttr_len.std()))
    loginfo_and_print(logger, "Vocabulary size: {}".format(len(word_count)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir",
        help="data directory",
        default="./data/rocstories"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "read_rocstories.log")
    format = "%(levelname)s : %(message)s"
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("data_dir: {}".format(data_dir))
    out_dir = "./data/rocstories/json"
    os.makedirs(out_dir, exist_ok=True)

    whole_data = []
    for data_name in ["train", "valid", "valid_right", "valid_wrong", "test"]:
        data = read_rocstories(
            os.path.join(data_dir, "rocstories.{}.csv".format(data_name.split("_")[0])),
            os.path.join(out_dir, "rocstories.{}.json".format(data_name))
        )
        if "_" not in data_name:
            whole_data.extend(data)

    analyze_corpus(whole_data, logger)
