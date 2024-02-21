# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import tqdm
import json
import csv
import glob
from collections import Counter
import numpy as np
import spacy

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger")
os.makedirs("./preprocessing/log", exist_ok=True)


def read_send(tsv_dir, json_path):
    filelist = glob.glob(os.path.join(tsv_dir, "*.tsv"))
    data = []
    for tsv_path in filelist:
        with open(tsv_path, "r") as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            tsv_data = [r for r in tsv_reader]
            data.append({
                "id": os.path.basename(tsv_path),
                "sentence": "".join([l[1] for l in tsv_data[1:]])
            })

    with open(json_path, "w") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    return data


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
        # default="./data/send/SENDv1_featuresRatings_pw/features"
        default="./data/send/featuresRatings/features"
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "read_send.log")
    format = "%(levelname)s : %(message)s"
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("data_dir: {}".format(data_dir))

    out_dir = "./data/send/json"
    os.makedirs(out_dir, exist_ok=True)

    whole_data = []
    for split in ["train", "valid", "test"]:
        whole_data.extend(read_send(
            os.path.join(data_dir, "{}/linguistic".format(split)),
            os.path.join(out_dir, "send.{}.json".format(split))
        ))

    analyze_corpus(whole_data, logger)
