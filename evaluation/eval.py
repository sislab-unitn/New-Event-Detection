# coding: utf-8

import os
import sys
from turtle import right

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import tqdm
import json
import csv

from utils.log_print import loginfo_and_print


logger = logging.getLogger("logger")
os.makedirs("./preprocessing/log", exist_ok=True)


def calc_point(context_data, candidate_data):
    point = 0
    pos_node_pairs = []
    neg_node_pairs = []
    pos_edge_pairs = []
    neg_edge_pairs = []

    for can_node in candidate_data["nodes"]:
        breaked = False
        con_node = None
        for con_node in context_data["nodes"]:
            for can_node_label in can_node["label"].split(", "):
                for con_node_label in con_node["label"].split(", "):
                    if can_node_label == con_node_label:
                        point += 1
                        breaked = True
                        pos_node_pairs.append((can_node, con_node))
                        break
            if breaked:
                break
        if not breaked:
            point -= 1
            neg_node_pairs.append((can_node, con_node))

    for can_edge in candidate_data["edges"]:
        breaked = False
        con_edge = None
        for con_edge in context_data["edges"]:
            if can_edge["root"] == con_edge["root"]:
                point += 1
                breaked = True
                pos_edge_pairs.append((can_edge, con_edge))
                break
        if not breaked:
            point -= 1
            neg_edge_pairs.append((can_edge, con_edge))

    return point, (pos_node_pairs, neg_node_pairs, pos_edge_pairs, neg_edge_pairs)


def eval(valid_path, right_path, wrong_path):
    with open(valid_path, "r") as f:
        valid_data = [json.loads(l.strip()) for l in f][0]

    with open(right_path, "r") as f:
        right_data = [json.loads(l.strip()) for l in f][0]

    with open(wrong_path, "r") as f:
        wrong_data = [json.loads(l.strip()) for l in f][0]

    num_correct = 0
    success_data = []
    error_data = []
    for v_d, r_d, w_d in zip(valid_data, right_data, wrong_data):
        assert v_d["id"] == r_d["id"] and r_d["id"] == w_d["id"]
        r_point, r_pairs = calc_point(v_d, r_d)
        w_point, w_pairs = calc_point(v_d, w_d)
        if r_point >= w_point:
            num_correct += 1
            success_data.append([r_point, w_point, v_d, r_d, w_d, r_pairs, w_pairs])
        else:
            error_data.append([r_point, w_point, v_d, r_d, w_d, r_pairs, w_pairs])

    num_data = len(valid_data)
    loginfo_and_print(logger, "Acc: {:.2f} ({}/{})".format(num_correct/num_data*100.0, num_correct, num_data))

    return success_data, error_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data",
        help="data (rocstories or send)",
        default="rocstories"
    )
    args = parser.parse_args()
    data = args.data
    
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, "eval.log")
    format = "%(levelname)s : %(message)s"
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("data: {}".format(data))
    if data != "rocstories" and data != "send":
        raise ValueError("Unknown dataset!")

    data_dir = "./data/{}/graph".format(data)

    valid_path = os.path.join(data_dir, "{}.valid.graph.json".format(data))
    right_path = os.path.join(data_dir, "{}.valid_right.graph.json".format(data))
    wrong_path = os.path.join(data_dir, "{}.valid_wrong.graph.json".format(data))
    success_data, error_data = eval(valid_path, right_path, wrong_path)

    with open(os.path.join(data_dir, "{}.valid.success.json".format(data)), "w") as f:
        for d in success_data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    with open(os.path.join(data_dir, "{}.valid.error.json".format(data)), "w") as f:
        for d in error_data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")
