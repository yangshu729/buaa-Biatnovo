from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import re

import numpy as np
import math
import argparse
import deepnovo_config as deepnovo_config


def read_feature_accuracy(input_file, split_char):
    feature_list = []
    with open(input_file, "r") as handle:
        header_line = handle.readline()
        for line in handle:
            line = re.split(split_char, line)
            feature = {}
            feature["feature_id"] = line[0]
            feature["feature_area"] = math.log10(float(line[1]))
            feature["predicted_score"] = float(line[4])
            feature["recall_AA"] = float(line[5])
            feature["predicted_len"] = float(line[6])
            feature_list.append(feature)
    return feature_list


def find_score_cutoff(accuracy_file, accuracy_cutoff):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80))  # section-separating line
    print("find_score_cutoff()")
    feature_list = read_feature_accuracy(accuracy_file, "\t|\r|\n")
    feature_list_sorted = sorted(feature_list, key=lambda k: k["predicted_score"], reverse=True)
    cutoff_index = int(len(feature_list) * accuracy_cutoff * 10 // 10)
    cutoff_score = feature_list_sorted[cutoff_index]["predicted_score"]
    print("cutoff_index = ", cutoff_index)
    print("cutoff_score = ", cutoff_score)
    print("cutoff_score = ", 100 * math.exp(cutoff_score))
    return cutoff_score


def select_top_score(input_file, output_file, split_char, col_score, score_cutoff):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80))  # section-separating line
    print("select_top_score()")
    print("input_file = ", input_file)
    print("output_file = ", output_file)
    print("score_cutoff = ", score_cutoff)
    total_feature = 0
    select_feature = 0
    with open(input_file, "r") as input_handle:
        with open(output_file, "w") as output_handle:
            header_line = input_handle.readline()
            predicted_list = []
            for line in input_handle:
                total_feature += 1
                line_split = re.split(split_char, line)
                predicted = {}
                predicted["line"] = line
                predicted["score"] = float(line_split[col_score]) if line_split[col_score] else -999
                if predicted["score"] >= score_cutoff:
                    select_feature += 1
                    print(predicted["line"], file=output_handle, end="")
    print("total_feature = ", total_feature)
    print("select_feature = ", select_feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ==============================================================================
    # FLAGS (options) for train
    # ==============================================================================
    parser.add_argument(
        "--test_top90", action="store_true", default=False, help="Set to True for test top 90% accuracy."
    )
    parser.add_argument("--denovo_file", type=str, default="", help="De Novo output file")
    parser.add_argument("--accuracy_file", type=str, default="", help="De Novo accuracy file.")
    opt = parser.parse_args()
    accuracy_cutoff = 0.1
    input_file = opt.denovo_file
    accuracy_file = opt.accuracy_file
    output_file = input_file + ".cut10"
    split_char = "\t|\n"
    col_score = deepnovo_config.pcol_score_max
    # col_score = deepnovo_config.pcol_score_max_unlabeled
    score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)
    select_top_score(input_file, output_file, split_char, col_score, score_cutoff)
