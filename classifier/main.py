# coding: utf-8

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import logging
import tqdm
from datetime import datetime
import argparse
import pickle
from math import sqrt

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification

from utils.log_print import loginfo_and_print

from classifier.train import run_epochs
from classifier.test import test
from classifier.dataset import Dataset
from classifier.model import NewEventDetectionClassifier

from classifier.baseline_test import baseline_test


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir",
        help="data directory",
        default="./data/send/annotation_result_iob"
    )
    parser.add_argument(
        "-b", "--bert_dir",
        help="bert directory",
        default="roberta-base"
    )
    parser.add_argument(
        "--method",
        help="classification method",
        default="method"
    )
    parser.add_argument(
        "-m", "--model_name",
        help="model file name",
        default="classifier.tar"
    )
    parser.add_argument(
        "--inference",
        help="inference mode",
        action="store_true"
    )
    parser.add_argument(
        "--use_valid",
        help="use valid data as test data",
        action="store_true"
    )

    hparams = parser.add_argument_group("hyper parameters",
        "model hyper parameters")
    hparams.add_argument(
        "--bert_size",
        help="bert hidden size",
        type=int,
        default=32000
    )
    hparams.add_argument(
        "--num_class",
        help="number of target class",
        type=int,
        default=2
    )
    hparams.add_argument(
        "--batch_size",
        help="batch size",
        type=int,
        default=16
    )
    hparams.add_argument(
        "--max_epoch",
        help="maximum epoch number",
        type=int,
        default=20
    )
    hparams.add_argument(
        "--max_gradient",
        help="maximum gradient",
        type=float,
        default=50.0
    )
    hparams.add_argument(
        "--learning_rate",
        help="learning rate",
        type=float,
        default=1e-4
    )
    hparams.add_argument(
        "--decay_step",
        help="decay step",
        type=int,
        default=200
    )
    hparams.add_argument(
        "--lr_decay",
        help="learning rate decay",
        type=float,
        default=0.5
    )
    hparams.add_argument(
        "--dropout",
        help="dropout probability",
        type=float,
        default=0.1
    )

    return parser


def parse_hparams(args):
    return {
        "SOS_id": 0,
        "PAD_id": 1,
        "EOS_id": 2,
        "UNK_id": 3,
        "MAX_UTTR_LEN": 30,

        "model_name": args.model_name,
        "method": args.method,
        "bert_name": os.path.basename(args.bert_dir),
        "bert_size": args.bert_size,
        "num_class": args.num_class,
        "batch_size": args.batch_size,
        "max_epoch": args.max_epoch,
        "max_gradient": args.max_gradient,
        "learning_rate": args.learning_rate,
        "decay_step": args.decay_step,
        "lr_decay": args.lr_decay,
        "dropout": args.dropout,
    }


def update_hparams(hparams):
    # hparams.update({
    #     "batch_size": 1,
    # })
    return hparams


if __name__ == "__main__":
    args = get_argparse().parse_args()
    data_dir = args.data_dir
    bert_dir = args.bert_dir
    method = args.method
    model_path = os.path.join("./save", "classifier."+args.model_name)
    train = not args.inference

    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(log_dir, args.model_name+".log")
    format = "%(message)s"
    logger = logging.getLogger("logger")
    handler = logging.FileHandler(filename=logfilename, mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not train:
        checkpoint = torch.load(model_path)
        hparams = checkpoint["hparams"]
        hparams = update_hparams(hparams)
    else:
        hparams = parse_hparams(args)

    for k, v in hparams.items():
        logger.info("{}: {}".format(k, v))

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)

    rule_based_method_list = [
        "random_one", "random_each", "first", "last", "subject", "entity"
    ]

    if method in rule_based_method_list:
        hparams["rule_based"] = True
    else:
        hparams["rule_based"] = False

    data_path = os.path.join(data_dir, "send.annotation.result.iob.20220728.json")
    print("Loading dataset...")
    dataset = Dataset(hparams, data_path, tokenizer)

    if method not in rule_based_method_list:
        def init_model():
            bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            return NewEventDetectionClassifier(hparams, bert_model, dataset.label_weight).cuda()

        print("Building model...")
        model = init_model()
        print("Model built and ready to go!")

    if hparams["rule_based"] == True:
        baseline_test(dataset, method)
    else:
        if train:
            print("Training model...")
            precision = 0
            recall = 0
            while True:
                precision, recall = run_epochs(hparams, model, dataset, model_path)
                if precision > 20.0 and recall > 20.0:
                    break
                print("Re-training model...")
                model = init_model()
            checkpoint = torch.load(model_path)

        print("Loading model...")
        model.load_state_dict(checkpoint["model"])

        print("Testing model...")
        dataset.switch_data("test")
        test(hparams, model, dataset)

    print("Done")
