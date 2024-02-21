# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import argparse
import json
from collections import defaultdict

from utils.log_print import loginfo_and_print


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "integrate_rule_results.log")
    format = "%(message)s"
    logger = logging.getLogger("logger")
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    filename_list = ["random", "beginning", "end"]

    for filename in filename_list:
        jsonfilename = os.path.join(log_dir, "{}.pasted.json".format(filename))
        if os.path.exists(jsonfilename):
            with open(os.path.join(jsonfilename), "r") as f:
                output = json.loads(f.read())
        if filename == "random":
            name = "Random"
        elif filename == "beginning":
            name = "Early"
        elif filename == "end":
            name = "Late"
        loginfo_and_print(logger, "\\textbf{{{}}} (pasted) & {:.1f} & {:.1f} & {:.1f} & {:2f} \\\\".format(
            name, output["precision"], output["recall"], output["f1"]*100, output["agg_f1"]
        ))

    for filename in filename_list:
        jsonfilename = os.path.join(log_dir, "{}.triplet_pasted.json".format(filename))
        if os.path.exists(jsonfilename):
            with open(os.path.join(jsonfilename), "r") as f:
                output = json.loads(f.read())
        if filename == "random":
            name = "Random"
        elif filename == "beginning":
            name = "Early"
        elif filename == "end":
            name = "Late"
        loginfo_and_print(logger, "\\textbf{{{}}} (triplet_pasted) & {:.1f} & {:.1f} & {:.1f} & {:2f} \\\\".format(
            name, output["precision"], output["recall"], output["f1"]*100, output["agg_f1"]
        ))
