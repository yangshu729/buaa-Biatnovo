# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

import re
import sys
import os

import numpy as np

# Get the parent directory's path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import deepnovo_config


class WorkerTest(object):
    def __init__(self, opt):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest.__init__()")

        # we currently use deepnovo_config to store both const & settings
        # the settings should be shown in __init__() to keep track carefully
        self.MZ_MAX = deepnovo_config.MZ_MAX

        self.target_file = opt.target_file
        self.predicted_file = opt.predict_file
        self.accuracy_file = opt.accuracy_file
        self.denovo_only_file = opt.denovo_only_file
        print("target_file = {0:s}".format(self.target_file))
        print("predicted_file = {0:s}".format(self.predicted_file))
        print("accuracy_file = {0:s}".format(self.accuracy_file))
        print("denovo_only_file = {0:s}".format(self.denovo_only_file))

        self.target_dict = {}
        self.predicted_list = []

        # 增加mass_matrix的初始化
        self.mass_matrix = np.load("mass_matrix.npy")

    def test_accuracy(self, db_peptide_list=None):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest.test_accuracy()")

        # write the accuracy of predicted peptides
        accuracy_handle = open(self.accuracy_file, "w")
        header_list = [
            "feature_id",
            "feature_area",
            "target_sequence",
            "predicted_sequence",
            "predicted_score",
            "recall_AA",
            "predicted_len",
            "target_len",
            "scan_list_middle",
            "scan_list_original",
        ]
        header_row = "\t".join(header_list)
        print(header_row, file=accuracy_handle, end="\n")

        # write denovo_only peptides
        denovo_only_handle = open(self.denovo_only_file, "w")
        header_list = [
            "feature_id",
            "feature_area",
            "predicted_sequence",
            "predicted_score",
            "predicted_score_max",
            "scan_list_middle",
            "scan_list_original",
        ]
        header_row = "\t".join(header_list)
        print(header_row, file=denovo_only_handle, end="\n")

        self._get_target()
        target_count_total = len(self.target_dict)
        target_len_total = sum([len(x) for x in self.target_dict.values()])

        # this part is tricky!
        # some target peptides are reported by PEAKS DB but not found in
        #   db_peptide_list due to mistakes in cleavage rules.
        # if db_peptide_list is given, we only consider those target peptides,
        #   otherwise, use all target peptides
        target_dict_db = {}
        if db_peptide_list is not None:
            for feature_id, target in self.target_dict.items():
                target_simplied = target
                # remove the extension 'mod' from variable modifications
                target_simplied = ["M" if x == "M(Oxidation)" else x for x in target_simplied]
                target_simplied = ["N" if x == "N(Deamidation)" else x for x in target_simplied]
                target_simplied = ["Q" if x == "Q(Deamidation)" else x for x in target_simplied]
                if target_simplied in db_peptide_list:
                    target_dict_db[feature_id] = target
                else:
                    print("target not found: ", target_simplied)
        else:
            target_dict_db = self.target_dict
        target_count_db = len(target_dict_db)
        target_len_db = sum([len(x) for x in target_dict_db.values()])

        # we also skip target peptides with precursor_mass > MZ_MAX
        target_dict_db_mass = {}
        for feature_id, peptide in target_dict_db.items():
            if self._compute_peptide_mass(peptide) <= self.MZ_MAX:
                target_dict_db_mass[feature_id] = peptide
        target_count_db_mass = len(target_dict_db_mass)
        target_len_db_mass = sum([len(x) for x in target_dict_db_mass.values()])

        # read predicted peptides from deepnovo or peaks
        self._get_predicted()

        # note that the prediction has already skipped precursor_mass > MZ_MAX
        # we also skip predicted peptides whose feature_id's are not in target_dict_db_mass
        predicted_count_mass = len(self.predicted_list)
        predicted_count_mass_db = 0
        predicted_len_mass_db = 0
        predicted_only = 0
        # the recall is calculated on remaining peptides
        recall_AA_total = 0.0
        recall_peptide_total = 0.0

        # record scan with multiple features
        scan_dict = {}
        cc = 0
        for index, predicted in enumerate(self.predicted_list):
            feature_id = predicted["feature_id"]
            feature_area = str(predicted["feature_area"])
            feature_scan_list_middle = predicted["scan_list_middle"]
            feature_scan_list_original = predicted["scan_list_original"]
            if feature_scan_list_original:
                for scan in re.split(";|\r|\n", feature_scan_list_original):
                    if scan in scan_dict:
                        scan_dict[scan]["feature_count"] += 1
                        scan_dict[scan]["feature_list"].append(feature_id)
                    else:
                        scan_dict[scan] = {}
                        scan_dict[scan]["feature_count"] = 1
                        scan_dict[scan]["feature_list"] = [feature_id]

            if feature_id in target_dict_db_mass:
                predicted_count_mass_db += 1

                target = target_dict_db_mass[feature_id]
                target_len = len(target)

                # if >= 1 denovo peptides reported, calculate the best accuracy, 如果一个feature有不止一个sequence
                best_recall_AA = 0
                best_predicted_sequence = predicted["sequence"][0]
                best_predicted_score = predicted["score"][0]

                for predicted_sequence, predicted_score in zip(predicted["sequence"], predicted["score"]):
                    predicted_AA_id = [deepnovo_config.vocab[x] for x in predicted_sequence]
                    target_AA_id = [deepnovo_config.vocab[x] for x in target]
                    recall_AA = self._match_AA_novor(target_AA_id, predicted_AA_id)
                    # print(predicted_sequence)
                    # print(target, recall_AA)
                    # print()
                    # cc += 1
                    # if cc == 100 : exit()
                    if recall_AA > best_recall_AA or (
                        recall_AA == best_recall_AA and predicted_score > best_predicted_score
                    ):
                        best_recall_AA = recall_AA
                        best_predicted_sequence = predicted_sequence[:]
                        best_predicted_score = predicted_score
                recall_AA = best_recall_AA
                predicted_sequence = best_predicted_sequence[:]
                predicted_score = best_predicted_score

                recall_AA_total += recall_AA
                if recall_AA == target_len:
                    recall_peptide_total += 1
                    # file = '/home2/wusiyu/DeepNovo_data/output.txt'
                    # with open(file, 'a+') as f:
                    #   line = "".join(target) + "\t" + "".join(predicted_sequence) +"\n"
                    #   f.write(line)
                predicted_len = len(predicted_sequence)
                predicted_len_mass_db += predicted_len

                # convert to string format to print out
                target_sequence = ",".join(target)
                predicted_sequence = ",".join(predicted_sequence)
                predicted_score = "{0:.2f}".format(predicted_score)
                recall_AA = "{0:d}".format(recall_AA)
                predicted_len = "{0:d}".format(predicted_len)
                target_len = "{0:d}".format(target_len)
                print_list = [
                    feature_id,
                    feature_area,
                    target_sequence,
                    predicted_sequence,
                    predicted_score,
                    recall_AA,
                    predicted_len,
                    target_len,
                    feature_scan_list_middle,
                    feature_scan_list_original,
                ]
                print_row = "\t".join(print_list)
                print(print_row, file=accuracy_handle, end="\n")
            else:
                predicted_only += 1
                predicted_sequence = ";".join([",".join(x) for x in predicted["sequence"]])
                predicted_score = ";".join(["{0:.2f}".format(x) for x in predicted["score"]])
                if predicted["score"]:
                    predicted_score_max = "{0:.2f}".format(np.max(predicted["score"]))
                else:
                    predicted_score_max = ""
                print_list = [
                    feature_id,
                    feature_area,
                    predicted_sequence,
                    predicted_score,
                    predicted_score_max,
                    feature_scan_list_middle,
                    feature_scan_list_original,
                ]
                print_row = "\t".join(print_list)
                print(print_row, file=denovo_only_handle, end="\n")

        accuracy_handle.close()
        denovo_only_handle.close()

        print("target_count_total = {0:d}".format(target_count_total))
        print("target_len_total = {0:d}".format(target_len_total))
        print("target_count_db = {0:d}".format(target_count_db))
        print("target_len_db = {0:d}".format(target_len_db))
        print("target_count_db_mass: {0:d}".format(target_count_db_mass))
        print("target_len_db_mass: {0:d}".format(target_len_db_mass))
        print()

        print("predicted_count_mass: {0:d}".format(predicted_count_mass))
        print("predicted_count_mass_db: {0:d}".format(predicted_count_mass_db))
        print("predicted_len_mass_db: {0:d}".format(predicted_len_mass_db))
        print("predicted_only: {0:d}".format(predicted_only))
        print(recall_AA_total)
        print("recall peptide total(correct peptide) = {0:.2f}".format(recall_peptide_total))
        print("recall_AA_total = {0:.4f}".format(recall_AA_total / target_len_total))
        print("recall_AA_db = {0:.4f}".format(recall_AA_total / target_len_db))
        print("recall_AA_db_mass = {0:.4f}".format(recall_AA_total / target_len_db_mass))
        print("recall_peptide_total = {0:.4f}".format(recall_peptide_total / target_count_total))
        print("recall_peptide_db = {0:.4f}".format(recall_peptide_total / target_count_db))
        print("recall_peptide_db_mass = {0:.4f}".format(recall_peptide_total / target_count_db_mass))
        print("precision_AA_mass_db  = {0:.4f}".format(recall_AA_total / predicted_len_mass_db))
        print("precision_peptide_mass_db  = {0:.4f}".format(recall_peptide_total / predicted_count_mass_db))

    def test_accuracy_position_bleu(self, db_peptide_list=None):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest.test_accuracy_position_bleu()")

        self._get_target()
        target_count_total = len(self.target_dict)
        target_len_total = sum([len(x) for x in self.target_dict.values()])

        target_dict_db = {}
        if db_peptide_list is not None:
            for feature_id, target in self.target_dict.items():
                target_simplied = target
                # remove the extension 'mod' from variable modifications
                target_simplied = ["M" if x == "M(Oxidation)" else x for x in target_simplied]
                target_simplied = ["N" if x == "N(Deamidation)" else x for x in target_simplied]
                target_simplied = ["Q" if x == "Q(Deamidation)" else x for x in target_simplied]

                if target_simplied in db_peptide_list:
                    target_dict_db[feature_id] = target
                else:
                    print("target not found: ", target_simplied)
        else:
            target_dict_db = self.target_dict
        target_count_db = len(target_dict_db)
        target_len_db = sum([len(x) for x in target_dict_db.values()])

        # we also skip target peptides with precursor_mass > MZ_MAX
        target_dict_db_mass = {}
        for feature_id, peptide in target_dict_db.items():
            if self._compute_peptide_mass(peptide) <= self.MZ_MAX:
                target_dict_db_mass[feature_id] = peptide
        target_count_db_mass = len(target_dict_db_mass)
        target_len_db_mass = sum([len(x) for x in target_dict_db_mass.values()])

        # read predicted peptides from deepnovo or peaks
        self._get_predicted()

        # note that the prediction has already skipped precursor_mass > MZ_MAX
        # we also skip predicted peptides whose feature_id's are not in target_dict_db_mass
        predicted_count_mass = len(self.predicted_list)
        predicted_count_mass_db = 0
        predicted_len_mass_db = 0
        predicted_only = 0
        # the recall is calculated on remaining peptides
        position_bleu_total = 0.0

        # record scan with multiple features
        scan_dict = {}
        cc = 0
        for index, predicted in enumerate(self.predicted_list):
            feature_id = predicted["feature_id"]

            feature_scan_list_original = predicted["scan_list_original"]
            if feature_scan_list_original:
                for scan in re.split(";|\r|\n", feature_scan_list_original):
                    if scan in scan_dict:
                        scan_dict[scan]["feature_count"] += 1
                        scan_dict[scan]["feature_list"].append(feature_id)
                    else:
                        scan_dict[scan] = {}
                        scan_dict[scan]["feature_count"] = 1
                        scan_dict[scan]["feature_list"] = [feature_id]

            if feature_id in target_dict_db_mass:
                predicted_count_mass_db += 1

                target = target_dict_db_mass[feature_id]
                target_len = len(target)

                for predicted_sequence, predicted_score in zip(predicted["sequence"], predicted["score"]):
                    # predicted_AA_id = [deepnovo_config.vocab[x] for x in predicted_sequence]
                    # target_AA_id = [deepnovo_config.vocab[x] for x in target]
                    predicted_AA_id = [x for x in predicted_sequence]
                    target_AA_id = [x for x in target]
                    position_bleu = self._cal_position_bleu(target_AA_id, predicted_AA_id)
                    # print(predicted_sequence)
                    # print(target, position_bleu)
                    # print()
                    # cc += 1
                    # if cc == 100: exit()

                position_bleu_total += position_bleu
                predicted_len = len(predicted_sequence)
                predicted_len_mass_db += predicted_len

            else:
                predicted_only += 1

        # print("target_count_total = {0:d}".format(target_count_total))
        # print("target_len_total = {0:d}".format(target_len_total))
        # print("target_count_db = {0:d}".format(target_count_db))
        # print("target_len_db = {0:d}".format(target_len_db))
        # print("target_count_db_mass: {0:d}".format(target_count_db_mass))
        # print("target_len_db_mass: {0:d}".format(target_len_db_mass))
        # print()
        #
        # print("predicted_count_mass: {0:d}".format(predicted_count_mass))
        print("predicted_count_mass_db: {0:d}".format(predicted_count_mass_db))
        print("predicted_len_mass_db: {0:d}".format(predicted_len_mass_db))
        # print("predicted_only: {0:d}".format(predicted_only))
        print("position_bleu_total = {0:.4f}".format(position_bleu_total))
        print("average position_bleu_total = {0:.4f}".format(position_bleu_total / target_count_total))
        print("average position_bleu_total_db_mass = {0:.4f}".format(position_bleu_total / target_count_db_mass))
        print("average position_bleu_predicted = {0:.4f}".format(position_bleu_total / predicted_count_mass_db))
        print("average position_bleu_predicted1 = {0:.4f}".format(position_bleu_total / predicted_len_mass_db))

    def test_accuracy_smith_waterman(self, db_peptide_list=None):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest.test_accuracy_position_bleu()")

        self._get_target()
        target_count_total = len(self.target_dict)
        target_len_total = sum([len(x) for x in self.target_dict.values()])

        target_dict_db = {}
        if db_peptide_list is not None:
            for feature_id, target in self.target_dict.items():
                target_simplied = target
                # remove the extension 'mod' from variable modifications
                target_simplied = ["M" if x == "M(Oxidation)" else x for x in target_simplied]
                target_simplied = ["N" if x == "N(Deamidation)" else x for x in target_simplied]
                target_simplied = ["Q" if x == "Q(Deamidation)" else x for x in target_simplied]
                if target_simplied in db_peptide_list:
                    target_dict_db[feature_id] = target
                else:
                    print("target not found: ", target_simplied)
        else:
            target_dict_db = self.target_dict
        target_count_db = len(target_dict_db)
        target_len_db = sum([len(x) for x in target_dict_db.values()])

        # we also skip target peptides with precursor_mass > MZ_MAX
        target_dict_db_mass = {}
        for feature_id, peptide in target_dict_db.items():
            if self._compute_peptide_mass(peptide) <= self.MZ_MAX:
                target_dict_db_mass[feature_id] = peptide
        target_count_db_mass = len(target_dict_db_mass)
        target_len_db_mass = sum([len(x) for x in target_dict_db_mass.values()])

        # read predicted peptides from deepnovo or peaks
        self._get_predicted()

        # note that the prediction has already skipped precursor_mass > MZ_MAX
        # we also skip predicted peptides whose feature_id's are not in target_dict_db_mass
        predicted_count_mass = len(self.predicted_list)
        predicted_count_mass_db = 0
        predicted_len_mass_db = 0
        predicted_only = 0
        # the recall is calculated on remaining peptides
        sw_score_total = 0.0

        # record scan with multiple features
        scan_dict = {}
        for index, predicted in enumerate(self.predicted_list):
            feature_id = predicted["feature_id"]

            feature_scan_list_original = predicted["scan_list_original"]
            if feature_scan_list_original:
                for scan in re.split(";|\r|\n", feature_scan_list_original):
                    if scan in scan_dict:
                        scan_dict[scan]["feature_count"] += 1
                        scan_dict[scan]["feature_list"].append(feature_id)
                    else:
                        scan_dict[scan] = {}
                        scan_dict[scan]["feature_count"] = 1
                        scan_dict[scan]["feature_list"] = [feature_id]

            if feature_id in target_dict_db_mass:
                predicted_count_mass_db += 1

                target = target_dict_db_mass[feature_id]
                target_len = len(target)

                for predicted_sequence, predicted_score in zip(predicted["sequence"], predicted["score"]):
                    # predicted_AA_id = [deepnovo_config.vocab[x] for x in predicted_sequence]
                    # target_AA_id = [deepnovo_config.vocab[x] for x in target]
                    predicted_AA_id = [deepnovo_config.vocab[x] - 3 for x in predicted_sequence]
                    target_AA_id = [deepnovo_config.vocab[x] - 3 for x in target]
                    sw_score = self._cal_smith_waterman(target_AA_id, predicted_AA_id, 1, 1)

                sw_score_total += sw_score
                predicted_len = len(predicted_sequence)
                predicted_len_mass_db += predicted_len

            else:
                predicted_only += 1

        print("sw_score_total = {0:.4f}".format(sw_score_total))
        print("average sw_score_total = {0:.4f}".format(sw_score_total / target_count_total))
        print("average sw_score_total_db_mass = {0:.4f}".format(sw_score_total / target_count_db_mass))
        print("average sw_score_predicted= {0:.4f}".format(sw_score_total / predicted_count_mass_db))
        print("average sw_score_predicted1= {0:.4f}".format(sw_score_total / predicted_len_mass_db))

    def _compute_peptide_mass(self, peptide):
        """TODO(nh2tran): docstring."""

        # ~ print("".join(["="] * 80)) # section-separating line ===
        # ~ print("WorkerDB: _compute_peptide_mass()")

        peptide_mass = (
            deepnovo_config.mass_N_terminus
            + sum(deepnovo_config.mass_AA[aa] for aa in peptide)
            + deepnovo_config.mass_C_terminus
        )

        return peptide_mass

    # 获取预测的肽段
    def _get_predicted(self):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest._get_predicted()")

        predicted_list = []
        col_feature_id = deepnovo_config.pcol_feature_id
        col_feature_area = deepnovo_config.pcol_feature_area
        col_sequence = deepnovo_config.pcol_sequence
        col_score = deepnovo_config.pcol_score
        col_scan_list_middle = deepnovo_config.pcol_scan_list_middle
        col_scan_list_original = deepnovo_config.pcol_scan_list_original
        with open(self.predicted_file, "r") as handle:
            # header
            handle.readline()
            for line in handle:
                line_split = re.split("\t|\n", line)
                predicted = {}
                predicted["feature_id"] = line_split[col_feature_id]
                predicted["feature_area"] = float(line_split[col_feature_area])
                predicted["scan_list_middle"] = line_split[col_scan_list_middle]
                predicted["scan_list_original"] = line_split[col_scan_list_original]
                if line_split[col_sequence]:  # not empty sequence
                    predicted["sequence"] = [re.split(",", x) for x in re.split(";", line_split[col_sequence])]
                    predicted["score"] = [float(x) for x in re.split(";", line_split[col_score])]
                else:
                    predicted["sequence"] = [[]]
                    predicted["score"] = [-999]
                predicted_list.append(predicted)

        self.predicted_list = predicted_list

    # 获取目标肽段
    def _get_target(self):
        """TODO(nh2tran): docstring."""

        print("".join(["="] * 80))  # section-separating line
        print("WorkerTest._get_target()")

        target_dict = {}
        with open(self.target_file, "r") as handle:
            header_line = handle.readline()
            for line in handle:
                line = re.split(",|\r|\n", line)
                feature_id = line[0]
                raw_sequence = line[deepnovo_config.col_raw_sequence]
                assert raw_sequence, "Error: wrong target format."
                peptide = self._parse_sequence(raw_sequence)
                target_dict[feature_id] = peptide
        self.target_dict = target_dict

    # 处理肽段中的修饰
    def _parse_sequence(self, raw_sequence):
        """TODO(nh2tran): docstring."""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerTest._parse_sequence()")

        raw_sequence_len = len(raw_sequence)
        peptide = []
        index = 0
        while index < raw_sequence_len:
            if raw_sequence[index] == "(":
                if peptide[-1] == "C" and raw_sequence[index : index + 8] == "(+57.02)":
                    peptide[-1] = "C(Carbamidomethylation)"
                    index += 8
                elif peptide[-1] == "M" and raw_sequence[index : index + 8] == "(+15.99)":
                    peptide[-1] = "M(Oxidation)"
                    index += 8
                elif peptide[-1] == "N" and raw_sequence[index : index + 6] == "(+.98)":
                    peptide[-1] = "N(Deamidation)"
                    index += 6
                elif peptide[-1] == "Q" and raw_sequence[index : index + 6] == "(+.98)":
                    peptide[-1] = "Q(Deamidation)"
                    index += 6
                else:  # unknown modification
                    print("ERROR: unknown modification!")
                    print("raw_sequence = ", raw_sequence)
                    sys.exit()
            else:
                peptide.append(raw_sequence[index])
                index += 1

        return peptide

    def _match_AA_novor(self, target, predicted):
        """TODO(nh2tran): docstring."""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerTest._test_AA_match_novor()")

        num_match = 0
        target_len = len(target)
        predicted_len = len(predicted)
        target_mass = [deepnovo_config.mass_ID[x] for x in target]
        target_mass_cum = np.cumsum(target_mass)
        predicted_mass = [deepnovo_config.mass_ID[x] for x in predicted]
        predicted_mass_cum = np.cumsum(predicted_mass)

        i = 0
        j = 0
        while i < target_len and j < predicted_len:
            # print(i, j)
            if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
                if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                    # ~ if  decoder_input[index_aa] == output[index_aa]:
                    num_match += 1
                i += 1
                j += 1
            elif target_mass_cum[i] < predicted_mass_cum[j]:
                i += 1
            else:
                j += 1

        return num_match

    def _cal_position_bleu(self, target, predicted):
        # 将I 都换成L
        # print(target)

        target = ["L" if i == "I" else i for i in target]
        predicted = ["L" if i == "I" else i for i in predicted]
        score_total = 0
        for i in range(len(predicted)):
            # 最大值是 1
            temp_i = predicted[i]
            min_dist = len(target) * 2
            for j in range(len(target)):
                # 如果两个字母相等
                if temp_i == target[j]:
                    dist = abs(j - i) + abs((len(target) - j) - (len(predicted) - i))
                    min_dist = min(min_dist, dist)

            if min_dist != len(target) * 2:
                score_total += 1 / (1 + min_dist)
                # print("i: ", i, "temp_i: ", temp_i,  "min dist: ", min_dist, "score: ", 1 / (1 + min_dist))

        return score_total

    def _cal_smith_waterman(self, target, predicted, u, v):
        a = len(target)
        b = len(predicted)
        S = np.zeros((a + 1, b + 1))
        L = np.zeros((a + 1, b + 1))
        P = np.zeros((a + 1, b + 1))
        Q = np.zeros((a + 1, b + 1))
        seq1 = [0] + target[:]
        seq2 = [0] + predicted[:]
        # print(seq1)
        # print(seq2)
        # print(self.mass_matrix)
        for r in range(1, b + 1 if a > b else a + 1):
            for c in range(r, b + 1):
                # print(seq1[r], seq2[c])
                L[r, c] = S[r - 1, c - 1] + self.mass_matrix[seq1[r], seq2[c]]
                P[r, c] = max(np.add(S[0:r, c], -(np.arange(r, 0, -1) * u + v)))
                Q[r, c] = max(np.add(S[r, 0:c], -(np.arange(c, 0, -1) * u + v)))
                S[r, c] = max([0, L[r, c], P[r, c], Q[r, c]])
            for c in range(r + 1, a + 1):
                L[c, r] = S[c - 1, r - 1] + self.mass_matrix[seq1[c], seq2[r]]
                P[c, r] = max(np.add(S[0:c, r], -(np.arange(c, 0, -1) * u + v)))
                Q[c, r] = max(np.add(S[c, 0:r], -(np.arange(r, 0, -1) * u + v)))
                S[c, r] = max([0, L[c, r], P[c, r], Q[c, r]])
        np.set_printoptions(threshold=np.inf)
        S = S.astype(np.int32)
        # print("S = ", S)
        # print(S.max())
        return S.max()
