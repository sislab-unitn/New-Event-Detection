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


class Dataset(Dataset):
    def __init__(self, hparams, data_path, tokenizer):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.read_data(data_path)
        self.switch_data("train")

    def read_data(self, path):
        pkl_file = path+".{}.classifier.pkl".format(self.hparams["bert_name"])

        if os.path.exists(pkl_file):
            print("Loading from pickle dump file...")
            with open(pkl_file, mode="rb") as f:
                data_dic = pickle.load(f)
            self.raw_train_data = data_dic["raw_train"]
            self.raw_valid_data = data_dic["raw_valid"]
            self.raw_test_data = data_dic["raw_test"]
            self.id_train_data = data_dic["id_train"]
            self.id_valid_data = data_dic["id_valid"]
            self.id_test_data = data_dic["id_test"]
            self.triplet_key_dic = data_dic["key_dic"]
            self.label_weight = data_dic["label_weight"]
            print("Done")
            return

        print("Reading files...")
        with open(path, "r") as f:
            data = json.loads(f.read())
        data, label_weight = self.format_data(data)
        self.label_weight = label_weight
        loginfo_and_print(logger, "Read {} narratives".format(len(data)))

        self.raw_train_data = []
        self.raw_valid_data = []
        self.raw_test_data = []
        for d in data:
            if d["split"] == "train":
                self.raw_train_data.append(d)
            elif d["split"] == "valid":
                self.raw_valid_data.append(d)
            else:
                self.raw_test_data.append(d)

        self.triplet_key_dic = {}
        self.id_train_data = self.convert_to_id_data(self.raw_train_data)
        self.id_valid_data = self.convert_to_id_data(self.raw_valid_data)
        self.id_test_data = self.convert_to_id_data(self.raw_test_data)

        print("Dump to pickle file...")
        data_dic = {
            "raw_train": self.raw_train_data,
            "raw_valid": self.raw_valid_data,
            "raw_test": self.raw_test_data,
            "id_train": self.id_train_data,
            "id_valid": self.id_valid_data,
            "id_test": self.id_test_data,
            "key_dic": self.triplet_key_dic,
            "label_weight": self.label_weight
        }
        with open(pkl_file, mode="wb") as f:
            pickle.dump(data_dic, f)
        print("Done")

    def format_data(self, data):
        data_dic = defaultdict(list)
        for batch in data:
            for narrative_key, narrative in batch.items():
                batch_name = narrative["annotations"]["batch"]
                if batch_name is None:
                    continue
                if "batch" not in batch_name:
                    continue
                data_dic[narrative_key].append(narrative)

        new_data = []
        num_positive = 0
        num_negative = 0
        for narrative_key, annotation_list in data_dic.items():
            narrative_data = {
                "narrative_id": narrative_key,
                "sentences": [],
                "triplets": [],
                "triplet_labels": [],
                "triplet_positions": [],
                "triplet_keys": [],
                "subjects": [],
                "predicates": [],
                "objects": [],
                "subject_coref_id": [],
                "object_coref_id": [],
                "split": annotation_list[0]["annotations"]["split"]
            }
            for sentence_key in annotation_list[0].keys():
                if sentence_key == "annotations":
                    continue
                narrative_data["sentences"].append(
                    annotation_list[0][sentence_key]["sentence"]
                )
                narrative_data["triplets"].append([])
                narrative_data["triplet_labels"].append([])
                narrative_data["triplet_positions"].append([])
                narrative_data["triplet_keys"].append([])
                narrative_data["subjects"].append([])
                narrative_data["predicates"].append([])
                narrative_data["objects"].append([])
                narrative_data["subject_coref_id"].append([])
                narrative_data["object_coref_id"].append([])
                triplets = annotation_list[0][sentence_key]["triplets"]
                triplet_count = defaultdict(int)
                for annotation in annotation_list:
                    correct_triplets = annotation[sentence_key].get(
                        "correct_triplets", [])
                    for triplet_key, triplet_dic in triplets.items():
                        triplet_count[triplet_key]
                        if triplet_key in correct_triplets:
                            triplet_count[triplet_key] += 1
                for triplet_key, count in triplet_count.items():
                    triplet_text = triplets[triplet_key]["triplet"]
                    triplet_text = triplet_text[1:len(triplet_text)-1]
                    triplet_text = re.sub(">", "", triplet_text)
                    triplet_components = triplet_text.split("] - [")
                    narrative_data["subjects"][-1].append(triplet_components[0])
                    narrative_data["predicates"][-1].append(triplet_components[1])
                    narrative_data["objects"][-1].append(triplet_components[2])
                    narrative_data["subject_coref_id"][-1].append(triplets[triplet_key]["subject_coref_id"])
                    narrative_data["object_coref_id"][-1].append(triplets[triplet_key]["object_coref_id"])

                    triplet_text = re.sub(
                        "( ->| -|\[|\])+", "", triplets[triplet_key]["triplet"]
                    )
                    narrative_data["triplets"][-1].append(triplet_text)
                    narrative_data["triplet_keys"][-1].append(triplet_key)

                    if len(annotation_list) == 5:
                        if count < 3:
                            narrative_data["triplet_labels"][-1].append(0)
                            if narrative_data["split"] != "test":
                                num_negative += 1
                        else:
                            narrative_data["triplet_labels"][-1].append(1)
                            if narrative_data["split"] != "test":
                                num_positive += 1
                    else:
                        if count < 1:
                            narrative_data["triplet_labels"][-1].append(0)
                            if narrative_data["split"] != "test":
                                num_negative += 1
                        else:
                            narrative_data["triplet_labels"][-1].append(1)
                            if narrative_data["split"] != "test":
                                num_positive += 1

                    subject_index = triplets[triplet_key]["subject_indices"][0]
                    predicate_index = triplets[triplet_key]["predicate_indices"][0]
                    object_index = triplets[triplet_key]["object_indices"][0]
                    narrative_data["triplet_positions"][-1].append(
                        min([subject_index, predicate_index, object_index])
                    )
            new_data.append(narrative_data)
        label_weight = [num_positive/(num_negative+num_positive), num_negative/(num_negative+num_positive)]

        return new_data, label_weight

    def convert_to_id_data(self, data):
        print("Converting to indexes...")
        # Convert to id
        id_data = []
        pbar = tqdm.tqdm(data, total=len(data))
        unk_count = 0
        token_count = 0

        def tokenize(text):
            text = re.sub(" \,", ",", text)
            text = re.sub(" \.", ".", text)
            text = re.sub(" '", "'", text)
            text = re.sub(" ;", ";", text)
            text = re.sub(" :", ":", text)
            res = self.tokenizer(text)
            return res["input_ids"], res["attention_mask"]

        def concat_context(previous_ids_list):
            if len(previous_ids_list) <= 0:
                return [1]
            previous_ids = previous_ids_list[-1][1:]
            for idx in range(len(previous_ids_list)-2, -1, -1):
                if len(previous_ids) > 400:
                    break
                previous_ids = previous_ids_list[idx][1:] + previous_ids
            return [0]+previous_ids

        for d in pbar:
            previous_sent_ids_list = []
            previous_sent_mask_list = []
            previous_new_event_ids_list = []
            previous_new_event_mask_list = []
            for sent_idx, (sentence, triplets, triplet_labels, triplet_keys) in enumerate(zip(d["sentences"], d["triplets"], d["triplet_labels"], d["triplet_keys"])):
                sent_ids, sent_mask = tokenize(sentence)
                for triplet, triplet_label, triplet_key in zip(triplets, triplet_labels, triplet_keys):
                    triplet_ids, triplet_mask = tokenize(triplet)
                    self.triplet_key_dic[triplet_key] = len(self.triplet_key_dic)
                    previous_sentences = concat_context(previous_sent_ids_list)
                    previous_sentences_mask = concat_context(previous_sent_mask_list)
                    previous_events = concat_context(previous_new_event_ids_list)
                    previous_events_mask = concat_context(previous_new_event_mask_list)
                    # src = triplet_ids + sent_ids[1:] + previous_sentences[1:] + previous_events[1:]
                    # src_mask = triplet_mask + sent_mask[1:] + previous_sentences_mask[1:] + previous_events_mask[1:]
                    src = triplet_ids + sent_ids[1:] + previous_events[1:]
                    src_mask = triplet_mask + sent_mask[1:] + previous_events_mask[1:]
                    id_data.append({
                        "triplet_key": [self.triplet_key_dic[triplet_key]],
                        "src": src,
                        "src_mask": src_mask,
                        "sentence": sent_ids,
                        "sentence_mask": sent_mask,
                        "triplet": triplet_ids,
                        "triplet_mask": triplet_mask,
                        "previous_sentences": previous_sentences,
                        "previous_sentences_mask": previous_sentences_mask,
                        "previous_events": previous_events,
                        "previous_events_mask": previous_events_mask,
                        "triplet_label": [triplet_label],
                    })

                    if triplet_label == 1:
                        previous_new_event_ids_list.append(triplet_ids)
                        previous_new_event_mask_list.append(triplet_mask)
                previous_sent_ids_list.append(sent_ids)
                previous_sent_mask_list.append(sent_mask)
                count = Counter(sent_ids)
                unk_count += count[self.hparams["UNK_id"]]
                token_count += len(sent_ids)

        print("UNK count: {:.2f} ({}/{})".format(unk_count/token_count*100, unk_count, token_count))

        return id_data

    def switch_data(self, type):
        if type == "valid":
            if self.hparams["rule_based"] == True:
                self.data = self.raw_valid_data
            else:
                self.data = self.id_valid_data
        elif type == "test":
            if self.hparams["rule_based"] == True:
                self.data = self.raw_test_data
            else:
                self.data = self.id_test_data
        else:
            if self.hparams["rule_based"] == True:
                self.data = self.raw_train_data
            else:
                self.data = self.id_train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v if type(v) is torch.Tensor else torch.tensor(v) for k, v in self.data[idx].items()}
