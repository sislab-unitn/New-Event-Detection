# coding: utf-8

# import random
import imp
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
        self.label_to_id = {"i": 1, "o": 0, "b": 2}
        self.read_data(data_path)
        self.switch_data("train")

    def read_data(self, path):
        if self.hparams["pasted_only"]:
            pkl_file = path+".{}.pasted.tagger.pkl".format(self.hparams["bert_name"])
        else:
            pkl_file = path+".{}.triplet_pasted.tagger.pkl".format(self.hparams["bert_name"])

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
            self.triplet_key_dic = data_dic["triplet_key_dic"]
            self.narrative_key_to_id_dict = data_dic["narrative_key_dic"]
            self.id_to_narrative_key_dic = data_dic["id_narrative_key_dic"]
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
        self.narrative_key_to_id_dict = {}
        self.id_train_data = self.convert_to_id_data(self.raw_train_data)
        self.id_valid_data = self.convert_to_id_data(self.raw_valid_data)
        self.id_test_data = self.convert_to_id_data(self.raw_test_data)
        self.id_to_narrative_key_dic = {id: key for key, id in self.narrative_key_to_id_dict.items()}

        print("Dump to pickle file...")
        data_dic = {
            "raw_train": self.raw_train_data,
            "raw_valid": self.raw_valid_data,
            "raw_test": self.raw_test_data,
            "id_train": self.id_train_data,
            "id_valid": self.id_valid_data,
            "id_test": self.id_test_data,
            "triplet_key_dic": self.triplet_key_dic,
            "narrative_key_dic": self.narrative_key_to_id_dict,
            "id_narrative_key_dic": self.id_to_narrative_key_dic,
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
                "narrative_key": narrative_key,
                "sentences": [],
                "iob_tag": [],
                "tagged_words": [],
                "sentence_keys": [],
                "split": annotation_list[0]["annotations"]["split"],
                "triplet_pasted_iob_tag": []
            }
            for sentence_key in annotation_list[0].keys():
                if sentence_key == "annotations":
                    continue
                narrative_data["sentence_keys"].append(
                    sentence_key
                )
                narrative_data["sentences"].append(
                    annotation_list[0][sentence_key]["sentence"]
                )
                iob_list = []
                for annotation in annotation_list:
                    if self.hparams["pasted_only"]:
                        iob_list.append(annotation[sentence_key]["pasted_iob"])
                    else:
                        iob_list.append(annotation[sentence_key]["triplet_pasted_iob"])
                iob_tag = []
                for idx in range(len(iob_list[0])):
                    tag = max(Counter([iob[idx] for iob in iob_list]).items(), key=lambda x:x[1])[0]
                    if tag == "b":
                        tag = "i"
                    # if tag == "i" and iob_tag[-1] == "o":
                    #     tag = "b"
                    iob_tag.append(tag)
                    if narrative_data["split"] != "test":
                        if tag == "i":
                            num_positive += 1
                        else:
                            num_negative += 1
                narrative_data["iob_tag"].append(iob_tag)

                # triplet_pasted test for model trained with copy-pasted
                iob_list = []
                for annotation in annotation_list:
                    iob_list.append(annotation[sentence_key]["triplet_pasted_iob"])
                iob_tag = []
                for idx in range(len(iob_list[0])):
                    tag = max(Counter([iob[idx] for iob in iob_list]).items(), key=lambda x:x[1])[0]
                    if tag == "b":
                        tag = "i"
                    iob_tag.append(tag)
                narrative_data["triplet_pasted_iob_tag"].append(iob_tag)

                tagged_words = []
                for word, tag in zip(narrative_data["sentences"][-1].split(" "), iob_tag):
                    if tag != "o":
                        tagged_words.append(word)
                narrative_data["tagged_words"].append(tagged_words)
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
            previous_tagged_words_ids_list = []
            previous_tagged_words_mask_list = []
            self.narrative_key_to_id_dict[d["narrative_key"]] = len(self.narrative_key_to_id_dict)
            for sent_idx, (sentence, iob_tag, tagged_words, sentence_key, triplet_pasted_iob_tag) in enumerate(zip(d["sentences"], d["iob_tag"], d["tagged_words"], d["sentence_keys"], d["triplet_pasted_iob_tag"])):
                self.narrative_key_to_id_dict[sentence_key] = len(self.narrative_key_to_id_dict)
                sent_ids, sent_mask, aligned_iob_tag, word_ids = self.tokenize_and_align_labels(sentence, iob_tag)
                tokenized_inputs = self.tokenizer(
                    tagged_words,
                    truncation=True,
                    # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                    is_split_into_words=True,
                )
                tagged_words_ids = tokenized_inputs["input_ids"]
                tagged_words_mask = tokenized_inputs["attention_mask"]

                previous_sentences = concat_context(previous_sent_ids_list)
                previous_sentences_mask = concat_context(previous_sent_mask_list)
                previous_tagged_words = concat_context(previous_tagged_words_ids_list)
                previous_tagged_words_mask = concat_context(previous_tagged_words_mask_list)

                src = sent_ids + previous_tagged_words[1:]
                src_mask = sent_mask + previous_tagged_words_mask[1:]
                id_data.append({
                    "narrative_key": [self.narrative_key_to_id_dict[d["narrative_key"]]],
                    "sentence_key": [self.narrative_key_to_id_dict[sentence_key]],
                    "src": src,
                    "src_mask": src_mask,
                    "iob_tag": aligned_iob_tag,
                    "sentence": sent_ids,
                    "previous_sentences": previous_sentences,
                    "previous_tagged_words": previous_tagged_words,
                    "word_ids": word_ids,
                    "original_iob_tag": [0 if tag == "o" else 1 for tag in iob_tag],
                    "original_triplet_pasted_iob_tag": [0 if tag == "o" else 1 for tag in triplet_pasted_iob_tag]
                })

                previous_sent_ids_list.append(sent_ids)
                previous_sent_mask_list.append(sent_mask)
                previous_tagged_words_ids_list.append(tagged_words_ids)
                previous_tagged_words_mask_list.append(tagged_words_mask)
                count = Counter(sent_ids)
                unk_count += count[self.hparams["UNK_id"]]
                token_count += len(sent_ids)

        print("UNK count: {:.2f} ({}/{})".format(unk_count/token_count*100, unk_count, token_count))

        return id_data

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(self, sentence, labels):
        tokenized_inputs = self.tokenizer(
            sentence.split(" "),
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        word_ids = tokenized_inputs.word_ids(0)
        previous_word_idx = None
        aligned_labels = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                aligned_labels.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label_to_id[labels[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                aligned_labels.append(self.label_to_id[labels[word_idx]])
            previous_word_idx = word_idx

        return tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], aligned_labels, word_ids[1:len(word_ids)-1]

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
