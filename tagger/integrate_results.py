# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import argparse
import json
from collections import defaultdict


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log_prefix",
        help="log file name prefix",
        required=True
    )

    return parser


if __name__ == "__main__":
    args = get_argparse().parse_args()
    log_prefix = args.log_prefix

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, os.path.splitext(os.path.basename(__file__))[0]+".{}.log".format(os.path.basename(log_prefix)))
    format = "%(message)s"
    logger = logging.getLogger("logger")
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    jsonfilename = log_prefix + ".tar.json"
    with open(os.path.join(jsonfilename), "r") as f:
        results = [json.loads(l.strip()) for l in f][0]

    output = defaultdict(list)
    jsonfilename = os.path.join(log_dir, os.path.splitext(os.path.basename(__file__))[0]+".{}.json".format(os.path.basename(log_prefix)))
    if os.path.exists(jsonfilename):
        with open(os.path.join(jsonfilename), "r") as f:
            output = [json.loads(l.strip()) for l in f][0]

    for key, value in results.items():
        output[key].append(value)
    with open(os.path.join(jsonfilename), "w") as f:
        json.dump(output, f, ensure_ascii=False)
        f.write("\n")
