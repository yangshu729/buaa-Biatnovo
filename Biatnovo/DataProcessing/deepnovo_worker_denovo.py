# encoding=utf-8
# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================


import os.path
import numpy as np
import torch
import deepnovo_config
import deepnovo_config_dda
from DataProcess.deepnovo_cython_modules import get_candidate_intensity
from DataProcess.deepnovo_cython_modules import get_candidate_intensity_dda
from DataProcessing.read import read_single_spectrum_true_feeding
import DataProcessing.read_dda as read_dda
from test_true_feeding import test_accuracy_score, test_accuracy_score_dda
from six.moves import xrange  # pylint: disable=redefined-builtin

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class WorkerDenovo(object):
    """TODO(nh2tran): docstring.
    This class contains the denovo sequencing module.
    """

    def __init__(self, type="DIA"):
        """TODO(nh2tran): docstring."""

        self.data_record = 0
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: __init__()")
        if type == "DIA":
            # we currently use deepnovo_config to store both const & settings
            # the settings should be shown in __init__() to keep track carefully
            self.neighbor_center = deepnovo_config.neighbor_size // 2  # 2
            self.knapsack_file = deepnovo_config.knapsack_file
            self.MZ_MAX = deepnovo_config.MZ_MAX
            self.mass_N_terminus = deepnovo_config.mass_N_terminus
            self.mass_C_terminus = deepnovo_config.mass_C_terminus
            self.KNAPSACK_AA_RESOLUTION = deepnovo_config.KNAPSACK_AA_RESOLUTION
            self.vocab_size = deepnovo_config.vocab_size
            self.GO_ID = deepnovo_config.GO_ID
            self.EOS_ID = deepnovo_config.EOS_ID
            self.mass_ID = deepnovo_config.mass_ID
            # 增加PAD_ID
            self.PAD_ID = deepnovo_config.PAD_ID
            self.precursor_mass_tolerance = deepnovo_config.precursor_mass_tolerance
            self.precursor_mass_ppm = deepnovo_config.precursor_mass_ppm
            self.num_position = deepnovo_config.num_position
            self.SPECTRUM_RESOLUTION = deepnovo_config.SPECTRUM_RESOLUTION
            self.mass_AA_min_round = deepnovo_config.mass_AA_min_round
            self.beam_size = deepnovo_config.beam_size
            self.vocab_reverse = deepnovo_config.vocab_reverse
            self.topk_output = deepnovo_config.topk_output
            print("knapsack_file = {0:s}".format(self.knapsack_file))
            # knapsack matrix will be loaded/built at the beginning of search_denovo()
            self.knapsack_matrix = None
            self.vocab = deepnovo_config.vocab
            self._buckets = deepnovo_config._buckets
            self.get_candidate_intensity = get_candidate_intensity
        elif type == "DDA":
            # we currently use deepnovo_config to store both const & settings
            # the settings should be shown in __init__() to keep track carefully
            self.neighbor_center = deepnovo_config_dda.neighbor_size // 2  # 2
            self.knapsack_file = deepnovo_config_dda.knapsack_file
            self.MZ_MAX = deepnovo_config_dda.MZ_MAX
            self.mass_N_terminus = deepnovo_config_dda.mass_N_terminus
            self.mass_C_terminus = deepnovo_config_dda.mass_C_terminus
            self.KNAPSACK_AA_RESOLUTION = deepnovo_config_dda.KNAPSACK_AA_RESOLUTION
            self.vocab_size = deepnovo_config_dda.vocab_size
            self.GO_ID = deepnovo_config_dda.GO_ID
            self.EOS_ID = deepnovo_config_dda.EOS_ID
            self.mass_ID = deepnovo_config_dda.mass_ID
            self.PAD_ID = deepnovo_config_dda.PAD_ID
            self.precursor_mass_tolerance = deepnovo_config_dda.precursor_mass_tolerance
            self.precursor_mass_ppm = deepnovo_config_dda.precursor_mass_ppm
            self.num_position = deepnovo_config_dda.num_position
            self.SPECTRUM_RESOLUTION = deepnovo_config_dda.SPECTRUM_RESOLUTION
            self.mass_AA_min_round = deepnovo_config_dda.mass_AA_min_round
            self.beam_size = deepnovo_config_dda.beam_size
            self.vocab_reverse = deepnovo_config_dda.vocab_reverse
            self.topk_output = deepnovo_config_dda.topk_output
            print("knapsack_file = {0:s}".format(self.knapsack_file))
            # knapsack matrix will be loaded/built at the beginning of search_denovo()
            self.knapsack_matrix = None
            self.vocab = deepnovo_config_dda.vocab
            self._buckets = deepnovo_config_dda._buckets
            self.get_candidate_intensity = get_candidate_intensity_dda

    def concate(self, forward_sequence, backward_sequence, precursor_mass):
        peptide_id_forward = [self.vocab[x] for x in forward_sequence]
        peptide_id_backward = [self.vocab[x] for x in backward_sequence]
        result = []
        begin_mass = precursor_mass - 22
        end_mass = precursor_mass - 18
        i = 0
        while i < len(forward_sequence):
            i += 1
            str_tmp = forward_sequence[:i]
            j = 0
            while j < len(backward_sequence) - 1:
                str_tmp1 = backward_sequence[j:]
                j += 1
                str_tmp2 = str_tmp + str_tmp1
                peptide_ids = [self.vocab[x] for x in str_tmp2]
                mass_tmp = sum(self.mass_ID[x] for x in peptide_ids)
                if mass_tmp < end_mass and mass_tmp > begin_mass:
                    result.append(str_tmp2)
        result = list(set([tuple(t) for t in result]))
        return result

    # 返回的结果包含预测本身
    def concate_more(self, forward_sequence, backward_sequence, precursor_mass):
        result = []
        begin_mass = precursor_mass - 22
        end_mass = precursor_mass - 18
        i = 0
        while i < len(forward_sequence):
            i += 1
            str_tmp = forward_sequence[:i]
            j = 0
            while j < len(backward_sequence) - 1:
                str_tmp1 = backward_sequence[j:]
                j += 1
                str_tmp2 = str_tmp + str_tmp1
                peptide_ids = [self.vocab[x] for x in str_tmp2]
                mass_tmp = sum(self.mass_ID[x] for x in peptide_ids)
                if mass_tmp < end_mass and mass_tmp > begin_mass:
                    result.append(str_tmp2)
        result.append(forward_sequence)
        result.append(backward_sequence)
        result = list(set([tuple(t) for t in result]))
        return result

    def search_denovo(self, model, worker_io):
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []
        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()
        worker_io.open_input()
        worker_io.get_location()
        worker_io.split_feature_index()
        worker_io.open_output()
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")
        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            predicted_batch = self._search_denovo_batch(spectrum_batch, model)
            predicted_denovo_list += predicted_batch
            worker_io.write_prediction(predicted_batch)

        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))
        worker_io.close_input()
        worker_io.close_output()
        return predicted_denovo_list

    def search_denovo_bi_indepedent(self, model, worker_io, opt, type="DIA"):
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []
        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()
        worker_io.open_input()
        worker_io.get_location()
        worker_io.split_feature_index()
        worker_io.open_output()
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")
        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            (
                predicted_batch_forward,
                predicted_batch_backward,
            ) = self._search_denovo_batch_bi_indepedent(spectrum_batch, model)
            spectrum_batch_size = len(spectrum_batch)
            data_set = [[] for _ in self._buckets]
            spectrum_index = [[] for _ in self._buckets]
            result_score = [[] for x in xrange(spectrum_batch_size)]
            # concatenate forward and backward paths
            for spectrum_id in xrange(spectrum_batch_size):
                if (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) > 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) > 0
                ):
                    concate_result = self.concate_more(
                        predicted_batch_forward[spectrum_id]["sequence"][0],
                        predicted_batch_backward[spectrum_id]["sequence"][0],
                        predicted_batch_forward[spectrum_id]["precursor_mz"]
                        * predicted_batch_forward[spectrum_id]["precursor_charge"],
                    )
                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) > 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) == 0
                ):
                    concate_result = predicted_batch_forward[spectrum_id]["sequence"]

                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) == 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) > 0
                ):
                    concate_result = predicted_batch_backward[spectrum_id]["sequence"]

                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) == 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) == 0
                ):
                    continue
                if type == "DIA":
                    result_list = [
                        read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                elif type == "DDA":
                    result_list = [
                        read_dda.read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                for result in result_list:
                    data, bucket_id, status = result
                    if data:
                        data_set[bucket_id].append(data)
                        spectrum_index[bucket_id].append(spectrum_id)  # 记录spectrum的index
            for bucket_id in xrange(len(self._buckets)):
                if data_set[bucket_id]:  # bucket not empty
                    print("test_set - bucket {0}".format(bucket_id))
                    if type == "DIA":
                        bi_score = test_accuracy_score(model, data_set, bucket_id, opt)
                    elif type == "DDA":
                        bi_score = test_accuracy_score_dda(model, data_set, bucket_id, opt)
                    index = 0
                    for i in bi_score:
                        result_score[spectrum_index[bucket_id][index]].append(i)
                        index += 1
            for spectrum_id in xrange(spectrum_batch_size):
                if len(result_score[spectrum_id]) > 0:
                    score_sum_list = []
                    for i in range(len(result_score[spectrum_id])):
                        score_tmp = result_score[spectrum_id][i]["score_sum"]
                        score_sum_list.append(score_tmp)
                    if len(score_sum_list) > 0:
                        tmp_i = score_sum_list.index(max(score_sum_list))
                        predicted_batch_forward[spectrum_id]["sequence"] = [
                            result_score[spectrum_id][tmp_i]["dec_string"]
                        ]
                        predicted_batch_forward[spectrum_id]["score"] = [result_score[spectrum_id][tmp_i]["score_sum"]]
                        predicted_batch_forward[spectrum_id]["position_score"] = [
                            result_score[spectrum_id][tmp_i]["position_score"]
                        ]
            worker_io.write_prediction(predicted_batch_forward)

        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))
        worker_io.close_input()
        worker_io.close_output()
        return predicted_denovo_list

    # 双向交互式生成
    def search_denovo_bi_SB(self, model, worker_io, opt, type="DIA"):
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []
        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()
        worker_io.open_input()
        worker_io.get_location()
        worker_io.split_feature_index()
        worker_io.open_output()
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")
        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            (
                predicted_batch_forward,
                predicted_batch_backward,
            ) = self._search_denovo_batch_bi_SB(spectrum_batch, model)
            spectrum_batch_size = len(spectrum_batch)
            data_set = [[] for _ in self._buckets]
            spectrum_index = [[] for _ in self._buckets]
            result_score = [[] for x in xrange(spectrum_batch_size)]
            for spectrum_id in xrange(spectrum_batch_size):
                if (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) > 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) > 0
                ):
                    concate_result = self.concate_more(
                        predicted_batch_forward[spectrum_id]["sequence"][0],
                        predicted_batch_backward[spectrum_id]["sequence"][0],
                        predicted_batch_forward[spectrum_id]["precursor_mz"]
                        * predicted_batch_forward[spectrum_id]["precursor_charge"],
                    )
                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) > 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) == 0
                ):
                    concate_result = predicted_batch_forward[spectrum_id]["sequence"]

                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) == 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) > 0
                ):
                    concate_result = predicted_batch_backward[spectrum_id]["sequence"]

                elif (
                    len(predicted_batch_forward[spectrum_id]["sequence"][0]) == 0
                    and len(predicted_batch_backward[spectrum_id]["sequence"][0]) == 0
                ):
                    continue
                if type == "DIA":
                    result_list = [
                        read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                elif type == "DDA":
                    result_list = [
                        read_dda.read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                for result in result_list:
                    data, bucket_id, status = result
                    if data:
                        data_set[bucket_id].append(data)
                        spectrum_index[bucket_id].append(spectrum_id)
            for bucket_id in xrange(len(self._buckets)):
                if data_set[bucket_id]:  # bucket not empty
                    print("test_set - bucket {0}".format(bucket_id))
                    if type == "DIA":
                        bi_score = test_accuracy_score(model, data_set, bucket_id, opt)
                    elif type == "DDA":
                        bi_score = test_accuracy_score_dda(model, data_set, bucket_id, opt)
                    index = 0
                    for i in bi_score:
                        result_score[spectrum_index[bucket_id][index]].append(i)
                        index += 1
            for spectrum_id in xrange(spectrum_batch_size):
                if len(result_score[spectrum_id]) > 0:
                    score_sum_list = []
                    for i in range(len(result_score[spectrum_id])):
                        score_tmp = result_score[spectrum_id][i]["score_sum"]
                        score_sum_list.append(score_tmp)
                    if len(score_sum_list) > 0:
                        tmp_i = score_sum_list.index(max(score_sum_list))
                        predicted_batch_forward[spectrum_id]["sequence"] = [
                            result_score[spectrum_id][tmp_i]["dec_string"]
                        ]
                        predicted_batch_forward[spectrum_id]["score"] = [result_score[spectrum_id][tmp_i]["score_sum"]]
                        predicted_batch_forward[spectrum_id]["position_score"] = [
                            result_score[spectrum_id][tmp_i]["position_score"]
                        ]
            worker_io.write_prediction(predicted_batch_forward)
        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))
        worker_io.close_input()
        worker_io.close_output()
        return predicted_denovo_list

    # 双向交互式生成,使用独立预测的前向反向结果
    def search_denovo_bi_SB_indepedent(self, model, model_ind, worker_io, opt):
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []

        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()

        worker_io.open_input()
        worker_io.get_location()
        worker_io.split_feature_index()
        worker_io.open_output()
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")
        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            (
                predicted_batch_forward,
                predicted_batch_backward,
            ) = self._search_denovo_batch_bi_SB(spectrum_batch, model)
            (
                predicted_batch_forward_indepedent,
                predicted_batch_backward_indepedent,
            ) = self._search_denovo_batch_bi_indepedent(spectrum_batch, model_ind)
            if (
                len(predicted_batch_forward) == 0
                and len(predicted_batch_backward) == 0
                and len(predicted_batch_forward_indepedent) == 0
                and len(predicted_batch_backward_indepedent) == 0
            ):
                continue
            if len(predicted_batch_forward[0]["score"]) > 0 and len(predicted_batch_backward[0]["score"]) > 0:
                concate_result = self.concate(
                    predicted_batch_forward[0]["sequence"][0],
                    predicted_batch_backward[0]["sequence"][0],
                    predicted_batch_forward[0]["precursor_mz"] * predicted_batch_forward[0]["precursor_charge"],
                )
                if (
                    len(predicted_batch_forward_indepedent[0]["score"]) > 0
                    and len(predicted_batch_backward_indepedent[0]["score"]) > 0
                ):
                    concate_result.append(tuple(predicted_batch_forward_indepedent[0]["sequence"][0]))
                    concate_result.append(tuple(predicted_batch_backward_indepedent[0]["sequence"][0]))
                elif len(predicted_batch_forward_indepedent[0]["score"]) > 0:
                    concate_result.append(tuple(predicted_batch_forward_indepedent[0]["sequence"][0]))
                elif len(predicted_batch_backward_indepedent[0]["score"]) > 0:
                    concate_result.append(tuple(predicted_batch_backward_indepedent[0]["sequence"][0]))
            else:
                if (
                    len(predicted_batch_forward_indepedent[0]["score"]) > 0
                    and len(predicted_batch_backward_indepedent[0]["score"]) > 0
                ):
                    concate_result = predicted_batch_forward_indepedent[0]["sequence"]
                    concate_result.append(tuple(predicted_batch_backward_indepedent[0]["sequence"][0]))
                elif len(predicted_batch_forward_indepedent[0]["score"]) > 0:
                    concate_result = predicted_batch_forward_indepedent[0]["sequence"]
                elif len(predicted_batch_backward_indepedent[0]["score"]) > 0:
                    concate_result = predicted_batch_backward_indepedent[0]["sequence"]
            if len(concate_result) > 0:
                if type == "DIA":
                    result_list = [
                        read_single_spectrum_true_feeding(spectrum_batch, sequence, opt) for sequence in concate_result
                    ]
                elif type == "DDA":
                    result_list = [
                        read_dda.read_single_spectrum_true_feeding(spectrum_batch, sequence, opt)
                        for sequence in concate_result
                    ]
                data_set = [[] for _ in self._buckets]
                bucket_feature = [[] for _ in self._buckets]
                index = 0
                for result in result_list:
                    if result is None:
                        index += 1
                        continue
                    data, bucket_id, status = result
                    if data:
                        data_set[bucket_id].append(data)
                        bucket_feature[bucket_id].append(index)  # 记录是第几个sequence
                score = []
                for bucket_id in xrange(len(self._buckets)):
                    if data_set[bucket_id]:  # bucket not empty
                        print("test_set - bucket {0}".format(bucket_id))
                        bi_score = test_accuracy_score(model_ind, data_set, bucket_id, bucket_feature, opt)
                        for i in range(len(bi_score)):
                            tmp = bi_score[i]
                            score.append(tmp)
                score_sum_list = []
                for i in range(len(score)):
                    score_tmp = score[i]["score_sum"]
                    score_sum_list.append(score_tmp)
                if len(score_sum_list) > 0:
                    tmp_i = score_sum_list.index(max(score_sum_list))
                    predicted_batch_forward[0]["sequence"] = [score[tmp_i]["dec_string"]]
                    predicted_batch_forward[0]["score"] = [score[tmp_i]["score_sum"]]
                    predicted_batch_forward[0]["position_score"] = [score[tmp_i]["position_score"]]
                    worker_io.write_prediction(predicted_batch_forward)

        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))

        worker_io.close_input()
        worker_io.close_output()
        return predicted_denovo_list

    # 采用单向forward和backward以及交互式backward和forward和预测拼接
    def search_denovo_bi_SB_indepedent_all(self, model, model_ind, worker_io, opt, type="DIA"):
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []
        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()

        worker_io.open_input()
        worker_io.get_location()
        # 分成batch
        worker_io.split_feature_index()
        worker_io.open_output()

        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")

        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            (
                predicted_batch_forward,
                predicted_batch_backward,
            ) = self._search_denovo_batch_bi_SB(spectrum_batch, model)
            (
                predicted_batch_forward_indepedent,
                predicted_batch_backward_indepedent,
            ) = self._search_denovo_batch_bi_indepedent(spectrum_batch, model_ind)
            spectrum_batch_size = len(spectrum_batch)
            data_set = [[] for _ in self._buckets]
            spectrum_index = [[] for _ in self._buckets]
            result_score = [[] for x in xrange(spectrum_batch_size)]
            # concatenate forward and backward paths
            for spectrum_id in xrange(spectrum_batch_size):
                if (
                    len(predicted_batch_forward) == 0
                    and len(predicted_batch_backward) == 0
                    and len(predicted_batch_forward_indepedent) == 0
                    and len(predicted_batch_backward_indepedent) == 0
                ):
                    continue
                # 交互式预测只有都输出才进行拼接
                if (
                    len(predicted_batch_forward[spectrum_id]["score"]) > 0
                    and len(predicted_batch_backward[spectrum_id]["score"]) > 0
                ):
                    concate_result = self.concate_more(
                        predicted_batch_forward[spectrum_id]["sequence"][0],
                        predicted_batch_backward[spectrum_id]["sequence"][0],
                        predicted_batch_forward[spectrum_id]["precursor_mz"]
                        * predicted_batch_forward[spectrum_id]["precursor_charge"],
                    )
                    if (
                        len(predicted_batch_forward_indepedent[spectrum_id]["score"]) > 0
                        and len(predicted_batch_backward_indepedent[spectrum_id]["score"]) > 0
                    ):
                        # 独立预测只有都输出才进行拼接
                        concate_result_inde = self.concate_more(
                            predicted_batch_forward_indepedent[spectrum_id]["sequence"][0],
                            predicted_batch_backward_indepedent[spectrum_id]["sequence"][0],
                            predicted_batch_forward_indepedent[spectrum_id]["precursor_mz"]
                            * predicted_batch_forward_indepedent[spectrum_id]["precursor_charge"],
                        )
                        for i in concate_result_inde:
                            concate_result.append(i)
                    elif len(predicted_batch_forward_indepedent[spectrum_id]["score"]) > 0:
                        concate_result.append(tuple(predicted_batch_forward_indepedent[spectrum_id]["sequence"][0]))
                    elif len(predicted_batch_backward_indepedent[spectrum_id]["score"]) > 0:
                        concate_result.append(tuple(predicted_batch_backward_indepedent[spectrum_id]["sequence"][0]))
                else:
                    if (
                        len(predicted_batch_forward_indepedent[spectrum_id]["score"]) > 0
                        and len(predicted_batch_backward_indepedent[spectrum_id]["score"]) > 0
                    ):
                        concate_result = self.concate_more(
                            predicted_batch_forward_indepedent[spectrum_id]["sequence"][0],
                            predicted_batch_backward_indepedent[spectrum_id]["sequence"][0],
                            predicted_batch_forward_indepedent[spectrum_id]["precursor_mz"]
                            * predicted_batch_forward_indepedent[spectrum_id]["precursor_charge"],
                        )
                    elif len(predicted_batch_forward_indepedent[spectrum_id]["score"]) > 0:
                        concate_result = predicted_batch_forward_indepedent[spectrum_id]["sequence"]
                    elif len(predicted_batch_backward_indepedent[spectrum_id]["score"]) > 0:
                        concate_result = predicted_batch_backward_indepedent[spectrum_id]["sequence"]

                if type == "DIA":
                    result_list = [
                        read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                elif type == "DDA":
                    result_list = [
                        read_dda.read_single_spectrum_true_feeding(spectrum_batch[spectrum_id], sequence, opt)
                        for sequence in concate_result
                    ]
                for result in result_list:
                    data, bucket_id, status = result
                    if data:
                        data_set[bucket_id].append(data)
                        spectrum_index[bucket_id].append(spectrum_id)

            for bucket_id in xrange(len(self._buckets)):
                if data_set[bucket_id]:  # bucket not empty
                    print("test_set - bucket {0}".format(bucket_id))
                    if type == "DIA":
                        bi_score = test_accuracy_score(model_ind, data_set, bucket_id, opt)
                    elif type == "DDA":
                        bi_score = test_accuracy_score_dda(model_ind, data_set, bucket_id, opt)
                    index = 0
                    for i in bi_score:
                        result_score[spectrum_index[bucket_id][index]].append(i)
                        index += 1
            for spectrum_id in xrange(spectrum_batch_size):
                if len(result_score[spectrum_id]) > 0:
                    score_sum_list = []
                    for i in range(len(result_score[spectrum_id])):
                        score_tmp = result_score[spectrum_id][i]["score_sum"]
                        score_sum_list.append(score_tmp)
                    if len(score_sum_list) > 0:
                        tmp_i = score_sum_list.index(max(score_sum_list))
                        predicted_batch_forward[spectrum_id]["sequence"] = [
                            result_score[spectrum_id][tmp_i]["dec_string"]
                        ]
                        predicted_batch_forward[spectrum_id]["score"] = [result_score[spectrum_id][tmp_i]["score_sum"]]
                        predicted_batch_forward[spectrum_id]["position_score"] = [
                            result_score[spectrum_id][tmp_i]["position_score"]
                        ]
            worker_io.write_prediction(predicted_batch_forward)
        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))

        worker_io.close_input()
        worker_io.close_output()

        return predicted_denovo_list

    # 采用单向forward和backward以及交互式backward和forward和预测拼接
    def search_denovo_bi_SB_indepedent_all_1(self, model, model_ind, worker_io, opt, type="DIA"):
        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo()")
        predicted_denovo_list = []

        # load/build knapsack matrix
        if os.path.isfile(self.knapsack_file):
            print("WorkerDenovo: search_denovo() - load knapsack matrix")
            self.knapsack_matrix = np.load(self.knapsack_file)
        else:
            print("WorkerDenovo: search_denovo() - build knapsack matrix")
            self.knapsack_matrix = self._build_knapsack()

        worker_io.open_input()
        worker_io.get_location()
        worker_io.split_feature_index()
        worker_io.open_output()

        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: search_denovo() - search loop")

        for index, feature_index_batch in enumerate(worker_io.feature_index_batch_list):
            print("Read {0:d}/{1:d} batches".format(index + 1, worker_io.feature_index_batch_count))
            spectrum_batch = worker_io.get_spectrum(feature_index_batch)
            (
                predicted_batch_forward,
                predicted_batch_backward,
            ) = self._search_denovo_batch_bi_SB(spectrum_batch, model)
            (
                predicted_batch_forward_indepedent,
                predicted_batch_backward_indepedent,
            ) = self._search_denovo_batch_bi_indepedent(spectrum_batch, model_ind)

            # 双向交互式预测进行拼接
            # 如果前向独立和后向独立都没有预测出来值，前向交互和反向交互都没有预测值
            if (
                len(predicted_batch_forward) == 0
                and len(predicted_batch_backward) == 0
                and len(predicted_batch_forward_indepedent) == 0
                and len(predicted_batch_backward_indepedent) == 0
            ):
                continue
            # 交互式预测只有都输出才进行拼接
            if len(predicted_batch_forward[0]["score"]) > 0 and len(predicted_batch_backward[0]["score"]) > 0:
                concate_result = self.concate_more(
                    predicted_batch_forward[0]["sequence"][0],
                    predicted_batch_backward[0]["sequence"][0],
                    predicted_batch_forward[0]["precursor_mz"] * predicted_batch_forward[0]["precursor_charge"],
                )
                if (
                    len(predicted_batch_forward_indepedent[0]["score"]) > 0
                    and len(predicted_batch_backward_indepedent[0]["score"]) > 0
                ):
                    # 独立预测只有都输出才进行拼接
                    concate_result_inde = self.concate_more(
                        predicted_batch_forward_indepedent[0]["sequence"][0],
                        predicted_batch_backward_indepedent[0]["sequence"][0],
                        predicted_batch_forward_indepedent[0]["precursor_mz"]
                        * predicted_batch_forward_indepedent[0]["precursor_charge"],
                    )
                    for i in concate_result_inde:
                        concate_result.append(i)
                elif len(predicted_batch_forward_indepedent[0]["score"]) > 0:
                    concate_result.append(tuple(predicted_batch_forward_indepedent[0]["sequence"][0]))
                elif len(predicted_batch_backward_indepedent[0]["score"]) > 0:
                    concate_result.append(tuple(predicted_batch_backward_indepedent[0]["sequence"][0]))
            else:
                if (
                    len(predicted_batch_forward_indepedent[0]["score"]) > 0
                    and len(predicted_batch_backward_indepedent[0]["score"]) > 0
                ):
                    concate_result = self.concate_more(
                        predicted_batch_forward_indepedent[0]["sequence"][0],
                        predicted_batch_backward_indepedent[0]["sequence"][0],
                        predicted_batch_forward_indepedent[0]["precursor_mz"]
                        * predicted_batch_forward_indepedent[0]["precursor_charge"],
                    )
                elif len(predicted_batch_forward_indepedent[0]["score"]) > 0:
                    concate_result = predicted_batch_forward_indepedent[0]["sequence"]
                elif len(predicted_batch_backward_indepedent[0]["score"]) > 0:
                    concate_result = predicted_batch_backward_indepedent[0]["sequence"]
            print("concate result:", concate_result)
            if len(concate_result) > 0:
                if type == "DIA":
                    result_list = [
                        read_single_spectrum_true_feeding(spectrum_batch[0], sequence, opt)
                        for sequence in concate_result
                    ]
                elif type == "DDA":
                    result_list = [
                        read_dda.read_single_spectrum_true_feeding(spectrum_batch[0], sequence, opt)
                        for sequence in concate_result
                    ]
                data_set = [[] for _ in self._buckets]
                bucket_feature = [[] for _ in self._buckets]
                index = 0
                for result in result_list:
                    if result is None:
                        index += 1
                        continue
                    data, bucket_id, status = result
                    if data:
                        data_set[bucket_id].append(data)
                        bucket_feature[bucket_id].append(index)

                score = []
                for bucket_id in xrange(len(self._buckets)):
                    if data_set[bucket_id]:  # bucket not empty
                        print("test_set - bucket {0}".format(bucket_id))
                        # test_accuracy(session, model, test_set, bucket_id)
                        if type == "DIA":
                            bi_score = test_accuracy_score(model_ind, data_set, bucket_id, opt)
                        elif type == "DDA":
                            bi_score = test_accuracy_score_dda(model_ind, data_set, bucket_id, opt)
                        for i in range(len(bi_score)):
                            tmp = bi_score[i]
                            score.append(tmp)

                score_sum_list = []
                for i in range(len(score)):
                    score_tmp = score[i]["score_sum"]
                    score_sum_list.append(score_tmp)
                print("score", score)
                print("score_sum_list:", score_sum_list)
                print("========================")
                if len(score_sum_list) > 0:
                    tmp_i = score_sum_list.index(max(score_sum_list))
                    predicted_batch_forward[0]["sequence"] = [score[tmp_i]["dec_string"]]
                    predicted_batch_forward[0]["score"] = [score[tmp_i]["score_sum"]]
                    predicted_batch_forward[0]["position_score"] = [score[tmp_i]["position_score"]]
                    worker_io.write_prediction(predicted_batch_forward)

        print("Total spectra: {0:d}".format(worker_io.feature_count["total"]))
        print("  read: {0:d}".format(worker_io.feature_count["read"]))
        print("  skipped: {0:d}".format(worker_io.feature_count["skipped"]))
        print("    by mass: {0:d}".format(worker_io.feature_count["skipped_mass"]))

        worker_io.close_input()
        worker_io.close_output()

        return predicted_denovo_list

    def _build_knapsack(self):
        """TODO(nh2tran): docstring.
        Build a static knapsack matrix by using dynamic programming.
        The knapsack matrix allows to retrieve all possible amino acids that
        could sum up to a given mass, subject to a given resolution.
        """

        print("".join(["="] * 80))  # section-separating line
        print("WorkerDenovo: _build_knapsack()")

        # maximum peptide mass, adjusted by the two terminals
        max_mass = self.MZ_MAX
        max_mass -= self.mass_N_terminus + self.mass_C_terminus
        # convert from float to integer as the algorithm only works with integer
        max_mass_round = int(round(max_mass * self.KNAPSACK_AA_RESOLUTION))
        # allow error tolerance up to 1 Dalton
        max_mass_upperbound = max_mass_round + self.KNAPSACK_AA_RESOLUTION

        knapsack_matrix = np.zeros(shape=(self.vocab_size, max_mass_upperbound), dtype=bool)

        # fill up the knapsack_matrix by rows and columns, using dynamic programming
        for AAid in xrange(3, self.vocab_size):  # excluding PAD, GO, EOS
            mass_AA = int(round(self.mass_ID[AAid] * self.KNAPSACK_AA_RESOLUTION))

            for col in xrange(max_mass_upperbound):
                current_mass = col + 1

                if current_mass < mass_AA:
                    knapsack_matrix[AAid, col] = False

                elif current_mass == mass_AA:
                    knapsack_matrix[AAid, col] = True

                elif current_mass > mass_AA:
                    sub_mass = current_mass - mass_AA
                    sub_col = sub_mass - 1
                    # check if the sub_mass can be formed by a combination of amino acids
                    # TODO(nh2tran): change np.sum to np.any
                    if np.sum(knapsack_matrix[:, sub_col]) > 0:
                        knapsack_matrix[AAid, col] = True
                        knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col], knapsack_matrix[:, sub_col])
                    else:
                        knapsack_matrix[AAid, col] = False

        np.save(self.knapsack_file, knapsack_matrix)
        return knapsack_matrix

    def _extend_peak(self, direction, model, spectrum_batch, peak_batch):
        print("WorkerDenovo: _extend_peak(), direction={0:s}".format(direction))

        spectrum_batch_size = len(spectrum_batch)
        top_path_batch = [[] for x in xrange(spectrum_batch_size)]

        # forward/backward direction setting
        #   the direction determines the model, the spectrum and the peak mass
        if direction == "forward":
            spectrum_original_name = "spectrum_original_forward"
            peak_mass_name = "prefix_mass"
            FIRST_LABEL = self.GO_ID
            LAST_LABEL = self.EOS_ID
        elif direction == "backward":
            spectrum_original_name = "spectrum_original_backward"
            peak_mass_name = "suffix_mass"
            FIRST_LABEL = self.EOS_ID
            LAST_LABEL = self.GO_ID
        spectrum_holder = torch.Tensor(np.array([x["spectrum_holder"] for x in spectrum_batch])).cuda()
        if spectrum_holder.size(0) == 0:
            return
        spectrum_cnn_outputs = model.Spectrum_output_inference(spectrum_holder)
        active_search_list = []
        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL]
            path["prefix_mass"] = peak_batch[spectrum_id][peak_mass_name]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list"] = [path]
            active_search_list.append(search_entry)  # batch个 path

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.
            # data blocks for the input feed of tensorflow model
            block_AAid_1 = []  # nobi
            block_AAid_2 = []  # nobi
            block_candidate_intensity = []
            # data blocks to record the current status of search entries
            block_AAid_list = []
            block_prefix_mass = []
            block_score_list = []
            block_score_sum = []
            block_knapsack_candidate = []

            # store the number of paths of each search entry in the big blocks
            #   to retrieve the info of each search entry later in STEP 4.
            search_entry_size = [0] * len(active_search_list)

            # gather data into blocks through 2 nested loops over active_search_list
            #   and over current_path_list of each search_entry
            for entry_index, search_entry in enumerate(active_search_list):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list = search_entry["current_path_list"]  # path列表
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name]  # (5, 150000)
                peak_mass_tolerance = peak_batch[spectrum_id]["mass_tolerance"]
                for path in current_path_list:
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        # AAid_1 = AAid_list[-2]
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        # AAid_1 = AAid_2
                        AAid_1 = [AAid_2]

                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]

                    if AAid_2 == LAST_LABEL:  # nobi
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )
                        continue

                    # get CANDIDATE INTENSITY to predict next AA
                    # TODO(nh2tran): change direction from 0/1 to "forward"/"backward"
                    direction_id = 0 if direction == "forward" else 1
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL]
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    if not knapsack_candidate:
                        knapsack_candidate.append(LAST_LABEL)

                    block_AAid_1.append(AAid_1)
                    block_AAid_2.append(AAid_2)
                    block_candidate_intensity.append(candidate_intensity)
                    block_AAid_list.append(AAid_list)
                    block_prefix_mass.append(prefix_mass)
                    block_score_list.append(score_list)
                    block_score_sum.append(score_sum)
                    block_knapsack_candidate.append(knapsack_candidate)

                    # record the size of each search entry in the blocks
                    search_entry_size[entry_index] += 1
            # STEP 3: run tensorflow model on data blocks to predict next AA.
            #   output is stored in current_log_prob, current_c_state, current_h_state
            if block_AAid_1:
                block_AAid_1 = np.array(block_AAid_1)  # nobi
                block_AAid_1 = torch.Tensor(block_AAid_1.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2 = torch.Tensor(np.array(block_AAid_2)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity = torch.Tensor(np.array(block_candidate_intensity)).cuda()
                if block_AAid_1.size(0) == 1 and block_AAid_1[0][0] == block_AAid_2[0][0]:
                    block_decoder_inputs = block_AAid_1
                else:
                    block_decoder_inputs = torch.cat([block_AAid_1, block_AAid_2], dim=0)
                with torch.no_grad():
                    candidate_size = block_decoder_inputs.size(1)
                    current_log_prob = []
                    for i in range(candidate_size):
                        decoder_inputs = block_decoder_inputs[:, i].unsqueeze_(1)  # (step - 1, 1)
                        candidate_intensity = block_candidate_intensity[i].unsqueeze_(0)
                        log_prob = model.Inference(
                            spectrum_cnn_outputs,
                            candidate_intensity,
                            decoder_inputs,
                            direction_id,
                        )
                        Logsoftmax = torch.nn.LogSoftmax(dim=1)
                        log_prob = Logsoftmax(log_prob)
                        current_log_prob.append(log_prob)
                    current_log_prob = torch.cat(current_log_prob, dim=0)
            # STEP 4: retrieve data from blocks to update the active_search_list
            #   with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list):
                # find all possible new paths within knapsack filter
                new_path_list = []
                for index in xrange(block_index, block_index + search_entry_size[entry_index]):
                    for AAid in block_knapsack_candidate[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            new_path["score_list"] = block_score_list[index] + [current_log_prob[index][AAid]]
                            new_path["score_sum"] = block_score_sum[index] + current_log_prob[index][AAid]
                        else:
                            new_path["score_list"] = block_score_list[index] + [0.0]
                            new_path["score_sum"] = block_score_sum[index] + 0.0
                        new_path_list.append(new_path)
                # beam search to select top candidates
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[: self.beam_size]
                    search_entry["current_path_list"] = [new_path_list[top_k_index[x]] for x in xrange(self.beam_size)]
                else:
                    search_entry["current_path_list"] = new_path_list
                block_index += search_entry_size[entry_index]

            # update active_search_list by removing empty entries
            active_search_list = [x for x in active_search_list if x["current_path_list"]]
            # STOP the extension loop if active_search_list is empty
            if not active_search_list:
                break
        return top_path_batch

    def _extend_peak_batch(self, direction, model, spectrum_batch, peak_batch):
        print("WorkerDenovo: _extend_peak(), direction={0:s}".format(direction))

        spectrum_batch_size = len(spectrum_batch)
        top_path_batch = [[] for x in xrange(spectrum_batch_size)]
        if direction == "forward":
            spectrum_original_name = "spectrum_original_forward"
            peak_mass_name = "prefix_mass"
            FIRST_LABEL = self.GO_ID
            LAST_LABEL = self.EOS_ID
        elif direction == "backward":
            spectrum_original_name = "spectrum_original_backward"
            peak_mass_name = "suffix_mass"
            FIRST_LABEL = self.EOS_ID
            LAST_LABEL = self.GO_ID
        spectrum_holder = torch.Tensor(np.array([x["spectrum_holder"] for x in spectrum_batch])).cuda()
        if spectrum_holder.size(0) == 0:
            return
        spectrum_cnn_outputs = model.Spectrum_output_inference(spectrum_holder)  # (32, 16, 128)
        active_search_list = []
        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL]
            path["prefix_mass"] = peak_batch[spectrum_id][peak_mass_name]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list"] = [path]
            active_search_list.append(search_entry)  # batch个 path

        self.data_record += 1
        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.
            block_AAid_1 = []  # nobi
            block_AAid_2 = []  # nobi
            block_candidate_intensity = []
            block_spectrum_cnn_outputs = []
            # data blocks to record the current status of search entries
            block_AAid_list = []
            block_prefix_mass = []
            block_score_list = []
            block_score_sum = []
            block_knapsack_candidate = []
            # store the number of paths of each search entry in the big blocks
            #   to retrieve the info of each search entry later in STEP 4.
            search_entry_size = [0] * len(active_search_list)

            # gather data into blocks through 2 nested loops over active_search_list
            #   and over current_path_list of each search_entry
            for entry_index, search_entry in enumerate(active_search_list):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list = search_entry["current_path_list"]
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name]  # (5, 150000)
                peak_mass_tolerance = peak_batch[spectrum_id]["mass_tolerance"]
                for path in current_path_list:
                    # keep track of the AA predicted in the previous iteration
                    #   for nobi (short k-mer) model, we will need 2 previous AA
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        # AAid_1 = AAid_list[-2]
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        AAid_1 = [AAid_2]
                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]
                    # when we reach LAST_LABEL, check if the mass of predicted sequence
                    #   is close to the given precursor_mass:
                    #   if yes, send the current path to output
                    #   if not, skip the current path
                    if AAid_2 == LAST_LABEL:  # nobi
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )
                        continue
                    # get CANDIDATE INTENSITY to predict next AA
                    # TODO(nh2tran): change direction from 0/1 to "forward"/"backward"
                    direction_id = 0 if direction == "forward" else 1
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL]
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    if not knapsack_candidate:
                        knapsack_candidate.append(LAST_LABEL)
                    block_AAid_1.append(AAid_1)
                    block_AAid_2.append(AAid_2)
                    block_candidate_intensity.append(candidate_intensity)
                    block_spectrum_cnn_outputs.append(spectrum_cnn_outputs[spectrum_id, :, :].unsqueeze_(0))
                    block_AAid_list.append(AAid_list)
                    block_prefix_mass.append(prefix_mass)
                    block_score_list.append(score_list)
                    block_score_sum.append(score_sum)
                    block_knapsack_candidate.append(knapsack_candidate)
                    # record the size of each search entry in the blocks
                    search_entry_size[entry_index] += 1
            # STEP 3: run tensorflow model on data blocks to predict next AA.
            #   output is stored in current_log_prob, current_c_state, current_h_state
            if block_AAid_1:
                block_AAid_1 = np.array(block_AAid_1)  # nobi
                block_AAid_1 = torch.Tensor(block_AAid_1.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2 = torch.Tensor(np.array(block_AAid_2)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity = torch.Tensor(np.array(block_candidate_intensity)).cuda()
                block_spectrum_cnn_outputs = torch.cat(block_spectrum_cnn_outputs, dim=0)
                if block_AAid_1.size(0) == 1 and block_AAid_1[0][0] == block_AAid_2[0][0]:
                    block_decoder_inputs = block_AAid_1
                else:
                    block_decoder_inputs = torch.cat([block_AAid_1, block_AAid_2], dim=0)
                with torch.no_grad():
                    log_prob = model.Inference(  # (batchsize, 26)
                        block_spectrum_cnn_outputs,
                        block_candidate_intensity,
                        block_decoder_inputs,
                        direction_id,
                    )
                    Logsoftmax = torch.nn.LogSoftmax(dim=1)
                    log_prob = Logsoftmax(log_prob)
                    current_log_prob = log_prob
            # STEP 4: retrieve data from blocks to update the active_search_list
            #   with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list):
                new_path_list = []
                for index in xrange(block_index, block_index + search_entry_size[entry_index]):
                    for AAid in block_knapsack_candidate[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            new_path["score_list"] = block_score_list[index] + [current_log_prob[index][AAid]]
                            new_path["score_sum"] = block_score_sum[index] + current_log_prob[index][AAid]
                        else:
                            new_path["score_list"] = block_score_list[index] + [0.0]
                            new_path["score_sum"] = block_score_sum[index] + 0.0
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[: self.beam_size]
                    search_entry["current_path_list"] = [new_path_list[top_k_index[x]] for x in xrange(self.beam_size)]
                else:
                    search_entry["current_path_list"] = new_path_list

                block_index += search_entry_size[entry_index]
            active_search_list = [x for x in active_search_list if x["current_path_list"]]
            # STOP the extension loop if active_search_list is empty
            if not active_search_list:
                break

        torch.cuda.empty_cache()
        return top_path_batch

    def _extend_peak_SB_batch(self, model, spectrum_batch, peak_batch):
        print("WorkerDenovo: _extend_peak(), direction={0:1}")

        spectrum_batch_size = len(spectrum_batch)
        top_path_batch_l2r = [[] for x in xrange(spectrum_batch_size)]
        top_path_batch_r2l = [[] for x in xrange(spectrum_batch_size)]

        spectrum_original_name_l2r = "spectrum_original_forward"
        peak_mass_name_l2r = "prefix_mass"
        FIRST_LABEL_l2r = self.GO_ID
        LAST_LABEL_l2r = self.EOS_ID
        spectrum_original_name_r2l = "spectrum_original_backward"
        peak_mass_name_r2l = "suffix_mass"
        FIRST_LABEL_r2l = self.EOS_ID
        LAST_LABEL_r2l = self.GO_ID
        PAD_LABEL = self.PAD_ID
        spectrum_holder = torch.Tensor(np.array([x["spectrum_holder"] for x in spectrum_batch])).cuda()

        if spectrum_holder.size(0) == 0:
            return [], []

        spectrum_cnn_outputs = model.Spectrum_output_inference(spectrum_holder)
        active_search_list_l2r = []
        active_search_list_r2l = []
        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL_l2r]
            path["prefix_mass"] = peak_batch[0][spectrum_id][peak_mass_name_l2r]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list_l2r"] = [path]
            active_search_list_l2r.append(search_entry)

        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL_r2l]
            path["prefix_mass"] = peak_batch[1][spectrum_id][peak_mass_name_r2l]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list_r2l"] = [path]
            active_search_list_r2l.append(search_entry)

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.
            block_AAid_1_l2r = []  # nobi
            block_AAid_2_l2r = []  # nobi
            block_candidate_intensity_l2r = []
            # data blocks to record the current status of search entries
            block_spectrum_cnn_outputs_l2r = []
            block_AAid_list_l2r = []
            block_prefix_mass_l2r = []
            block_score_list_l2r = []
            block_score_sum_l2r = []
            block_knapsack_candidate_l2r = []
            block_AAid_1_r2l = []  # nobi
            block_AAid_2_r2l = []  # nobi
            block_candidate_intensity_r2l = []
            # data blocks to record the current status of search entries
            block_spectrum_cnn_outputs_r2l = []
            block_AAid_list_r2l = []
            block_prefix_mass_r2l = []
            block_score_list_r2l = []
            block_score_sum_r2l = []
            block_knapsack_candidate_r2l = []

            # store the number of paths of each search entry in the big blocks
            #   to retrieve the info of each search entry later in STEP 4.
            search_entry_size_l2r = [0] * len(active_search_list_l2r)
            search_entry_size_r2l = [0] * len(active_search_list_r2l)
            block_spectrum_id_l2r = []
            block_spectrum_id_r2l = []
            for entry_index, search_entry in enumerate(active_search_list_l2r):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list_l2r = search_entry["current_path_list_l2r"]
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name_l2r]  # (5, 150000)
                peak_mass_tolerance = peak_batch[0][spectrum_id]["mass_tolerance"]
                for path in current_path_list_l2r:
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        AAid_1 = [AAid_2]

                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]
                    if AAid_2 == LAST_LABEL_l2r:  # nobi
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch_l2r[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )

                    # get CANDIDATE INTENSITY to predict next AA
                    direction_id = 0
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    # use knapsack and SUFFIX MASS to filter next AA candidate
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL_l2r]
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    # if not possible to extend, add LAST_LABEL to end
                    if not knapsack_candidate and AAid_2 != LAST_LABEL_l2r and AAid_2 != PAD_LABEL:
                        knapsack_candidate.append(LAST_LABEL_l2r)

                    if not knapsack_candidate and (AAid_2 == LAST_LABEL_l2r or AAid_2 == PAD_LABEL):
                        knapsack_candidate.append(PAD_LABEL)
                    # gather data blocks
                    block_AAid_1_l2r.append(AAid_1)
                    block_AAid_2_l2r.append(AAid_2)
                    block_candidate_intensity_l2r.append(candidate_intensity)
                    block_spectrum_cnn_outputs_l2r.append(spectrum_cnn_outputs[spectrum_id, :, :].unsqueeze_(0))
                    block_spectrum_id_l2r.append(spectrum_id)
                    block_AAid_list_l2r.append(AAid_list)
                    block_prefix_mass_l2r.append(prefix_mass)
                    block_score_list_l2r.append(score_list)
                    block_score_sum_l2r.append(score_sum)
                    block_knapsack_candidate_l2r.append(knapsack_candidate)

                    # record the size of each search entry in the blocks
                    search_entry_size_l2r[entry_index] += 1
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list_r2l = search_entry["current_path_list_r2l"]
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name_r2l]
                peak_mass_tolerance = peak_batch[1][spectrum_id]["mass_tolerance"]
                for path in current_path_list_r2l:
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        AAid_1 = [AAid_2]
                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]
                    if AAid_2 == LAST_LABEL_r2l:  # nobi
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch_r2l[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )

                    # get CANDIDATE INTENSITY to predict next AA
                    direction_id = 1
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    # use knapsack and SUFFIX MASS to filter next AA candidate
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL_r2l]
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    # if not possible to extend, add LAST_LABEL to end the sequence
                    if not knapsack_candidate and AAid_2 != LAST_LABEL_r2l and AAid_2 != PAD_LABEL:
                        knapsack_candidate.append(LAST_LABEL_r2l)

                    if not knapsack_candidate and (AAid_2 == LAST_LABEL_r2l or AAid_2 == PAD_LABEL):
                        knapsack_candidate.append(PAD_LABEL)
                    # gather data blocks
                    block_AAid_1_r2l.append(AAid_1)
                    block_AAid_2_r2l.append(AAid_2)
                    block_candidate_intensity_r2l.append(candidate_intensity)
                    block_spectrum_cnn_outputs_r2l.append(spectrum_cnn_outputs[spectrum_id, :, :].unsqueeze_(0))
                    block_spectrum_id_r2l.append(spectrum_id)
                    block_AAid_list_r2l.append(AAid_list)
                    block_prefix_mass_r2l.append(prefix_mass)
                    block_score_list_r2l.append(score_list)
                    block_score_sum_r2l.append(score_sum)
                    block_knapsack_candidate_r2l.append(knapsack_candidate)

                    # record the size of each search entry in the blocks
                    search_entry_size_r2l[entry_index] += 1
            # STEP 3: run tensorflow model on data blocks to predict next AA.
            #   output is stored in current_log_prob, current_c_state, current_h_state
            flag_l2r = [LAST_LABEL_l2r in i for i in block_AAid_list_l2r]
            flag_r2l = [LAST_LABEL_r2l in i for i in block_AAid_list_r2l]
            if not all(flag_l2r) or not all(flag_r2l):
                block_AAid_1_l2r = np.array(block_AAid_1_l2r)  # nobi
                block_AAid_1_l2r = torch.Tensor(block_AAid_1_l2r.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2_l2r = torch.Tensor(np.array(block_AAid_2_l2r)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity_l2r = torch.Tensor(np.array(block_candidate_intensity_l2r)).cuda()
                block_spectrum_cnn_outputs_l2r = torch.cat(block_spectrum_cnn_outputs_l2r, dim=0)
                block_AAid_1_r2l = np.array(block_AAid_1_r2l)  # nobi
                block_AAid_1_r2l = torch.Tensor(block_AAid_1_r2l.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2_r2l = torch.Tensor(np.array(block_AAid_2_r2l)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity_r2l = torch.Tensor(np.array(block_candidate_intensity_r2l)).cuda()
                block_spectrum_cnn_outputs_r2l = torch.cat(block_spectrum_cnn_outputs_r2l, dim=0)
                if block_AAid_1_l2r.size(0) == 1 and block_AAid_1_l2r[0][0] == block_AAid_2_l2r[0][0]:
                    block_decoder_inputs_l2r = block_AAid_1_l2r
                else:
                    block_decoder_inputs_l2r = torch.cat([block_AAid_1_l2r, block_AAid_2_l2r], dim=0)

                if block_AAid_1_r2l.size(0) == 1 and block_AAid_1_r2l[0][0] == block_AAid_2_r2l[0][0]:
                    block_decoder_inputs_r2l = block_AAid_1_r2l
                else:
                    block_decoder_inputs_r2l = torch.cat([block_AAid_1_r2l, block_AAid_2_r2l], dim=0)
                with torch.no_grad():
                    candidate_size_l2r = block_decoder_inputs_l2r.size(1)
                    candidate_size_r2l = block_decoder_inputs_r2l.size(1)
                    current_log_prob_l2r = []
                    current_log_prob_r2l = []
                    block_l2r = [
                        [i, j, k, l]
                        for i, j, k, l in zip(
                            block_spectrum_id_l2r,
                            block_spectrum_cnn_outputs_l2r,
                            block_candidate_intensity_l2r,
                            [block_decoder_inputs_l2r[:, f] for f in range(block_decoder_inputs_l2r.size(1))],
                        )
                    ]
                    block_r2l = [
                        [i, j, k, l]
                        for i, j, k, l in zip(
                            block_spectrum_id_r2l,
                            block_spectrum_cnn_outputs_r2l,
                            block_candidate_intensity_r2l,
                            [block_decoder_inputs_r2l[:, f] for f in range(block_decoder_inputs_r2l.size(1))],
                        )
                    ]
                    spectrum_ids = sorted(list(set(block_spectrum_id_r2l + block_spectrum_id_l2r)))
                    block_real_l2r = []
                    block_real_r2l = []
                    for spectrum_id in spectrum_ids:
                        block_l2r_per = [i for i in block_l2r if i[0] == spectrum_id]
                        block_r2l_per = [i for i in block_r2l if i[0] == spectrum_id]
                        len_min = min(len(block_l2r_per), len(block_r2l_per))
                        block_l2r_per = block_l2r_per[:len_min]
                        block_r2l_per = block_r2l_per[:len_min]
                        block_real_l2r.extend(block_l2r_per)
                        block_real_r2l.extend(block_r2l_per)
                    b_spectrum_cnn_outputs = torch.stack([i[1] for i in block_real_l2r], dim=0)
                    b_candidate_intensity_l2r = torch.stack([i[2] for i in block_real_l2r], dim=0)
                    b_candidate_intensity_r2l = torch.stack([i[2] for i in block_real_r2l], dim=0)
                    b_decoder_inputs_l2r = torch.stack([i[3] for i in block_real_l2r], dim=1)
                    b_decoder_inputs_r2l = torch.stack([i[3] for i in block_real_r2l], dim=1)
                    b_specturm_id = [i[0] for i in block_real_l2r]
                    log_prob_l2r, log_prob_r2l = model.Inference_SB(
                        b_spectrum_cnn_outputs,
                        b_candidate_intensity_l2r,
                        b_candidate_intensity_r2l,
                        b_decoder_inputs_l2r,
                        b_decoder_inputs_r2l,
                    )
                    Logsoftmax = torch.nn.LogSoftmax(dim=1)
                    log_prob_l2r = Logsoftmax(log_prob_l2r)
                    current_log_prob_l2r = log_prob_l2r
                    log_prob_r2l = Logsoftmax(log_prob_r2l)
                    current_log_prob_r2l = log_prob_r2l
            if all(flag_l2r) and all(flag_r2l):
                break
            # STEP 4: retrieve data from blocks to update the active_search_list
            #   with knapsack dynamic programming and beam search.
            block_index_l2r = 0
            for entry_index, search_entry in enumerate(active_search_list_l2r):
                # find all possible new paths within knapsack filter
                spectrum_id = search_entry["spectrum_id"]
                new_path_list = []
                for index in xrange(
                    block_index_l2r,
                    block_index_l2r + search_entry_size_l2r[entry_index],
                ):
                    for AAid in block_knapsack_candidate_l2r[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list_l2r[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass_l2r[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            try:
                                new_path["score_list"] = block_score_list_l2r[index] + [
                                    current_log_prob_l2r[index][AAid]
                                ]
                                new_path["score_sum"] = block_score_sum_l2r[index] + current_log_prob_l2r[index][AAid]
                            except Exception as inst:
                                import pdb

                                pdb.set_trace()
                        else:
                            new_path["score_list"] = block_score_list_l2r[index] + [0.0]
                            new_path["score_sum"] = block_score_sum_l2r[index] + 0.0
                        new_path_list.append(new_path)
                # beam search to select top candidates
                if len(new_path_list) > self.beam_size // 2:
                    try:
                        new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                        top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]
                        search_entry["current_path_list_l2r"] = [
                            new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                        ]
                    except Exception as inst:
                        new_path_score = np.array([x["score_sum"] for x in new_path_list])
                        top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]

                        search_entry["current_path_list_l2r"] = [
                            new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                        ]
                else:
                    search_entry["current_path_list_l2r"] = new_path_list
                    search_entry["current_path_list_l2r"].sort(key=lambda t: t["score_sum"], reverse=True)

                block_index_l2r += search_entry_size_l2r[entry_index]

            active_search_list_l2r = [x for x in active_search_list_l2r if x["current_path_list_l2r"]]

            block_index_r2l = 0
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                spectrum_id = search_entry["spectrum_id"]
                current_log_prob_r2l_specturm = [
                    i for i, j in zip(current_log_prob_r2l, b_specturm_id) if j == spectrum_id
                ]
                # find all possible new paths within knapsack filter
                new_path_list = []
                for index in xrange(
                    block_index_r2l,
                    block_index_r2l + search_entry_size_r2l[entry_index],
                ):
                    for AAid in block_knapsack_candidate_r2l[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list_r2l[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass_r2l[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            new_path["score_list"] = block_score_list_r2l[index] + [current_log_prob_r2l[index][AAid]]
                            new_path["score_sum"] = block_score_sum_r2l[index] + current_log_prob_r2l[index][AAid]
                        else:
                            new_path["score_list"] = block_score_list_r2l[index] + [0.0]
                            new_path["score_sum"] = block_score_sum_r2l[index] + 0.0
                        new_path_list.append(new_path)
                # beam search to select top candidates
                if len(new_path_list) > self.beam_size // 2:
                    try:
                        new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                        top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]
                        search_entry["current_path_list_r2l"] = [
                            new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                        ]
                    except Exception as inst:
                        new_path_score = np.array([x["score_sum"] for x in new_path_list])
                        top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]
                        search_entry["current_path_list_r2l"] = [
                            new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                        ]
                else:
                    search_entry["current_path_list_r2l"] = new_path_list
                    search_entry["current_path_list_r2l"].sort(key=lambda t: t["score_sum"], reverse=True)
                block_index_r2l += search_entry_size_r2l[entry_index]

            active_search_list_r2l = [x for x in active_search_list_r2l if x["current_path_list_r2l"]]

        return top_path_batch_l2r, top_path_batch_r2l

    def _extend_peak_SB(self, model, spectrum_batch, peak_batch):
        print("WorkerDenovo: _extend_peak(), direction={0:1}")

        spectrum_batch_size = len(spectrum_batch)
        top_path_batch_l2r = [[] for x in xrange(spectrum_batch_size)]
        top_path_batch_r2l = [[] for x in xrange(spectrum_batch_size)]

        spectrum_original_name_l2r = "spectrum_original_forward"
        peak_mass_name_l2r = "prefix_mass"
        FIRST_LABEL_l2r = self.GO_ID
        LAST_LABEL_l2r = self.EOS_ID
        spectrum_original_name_r2l = "spectrum_original_backward"
        peak_mass_name_r2l = "suffix_mass"
        FIRST_LABEL_r2l = self.EOS_ID
        LAST_LABEL_r2l = self.GO_ID
        PAD_LABEL = self.PAD_ID
        spectrum_holder = torch.Tensor(np.array([x["spectrum_holder"] for x in spectrum_batch])).cuda()

        if spectrum_holder.size(0) == 0:
            return [], []
        spectrum_cnn_outputs = model.Spectrum_output_inference(spectrum_holder)
        active_search_list_l2r = []
        active_search_list_r2l = []
        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL_l2r]
            path["prefix_mass"] = peak_batch[0][spectrum_id][peak_mass_name_l2r]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list_l2r"] = [path]
            active_search_list_l2r.append(search_entry)

        for spectrum_id in xrange(spectrum_batch_size):
            search_entry = {}
            search_entry["spectrum_id"] = spectrum_id
            path = {}
            path["AAid_list"] = [FIRST_LABEL_r2l]
            path["prefix_mass"] = peak_batch[1][spectrum_id][peak_mass_name_r2l]
            path["score_list"] = [0.0]
            path["score_sum"] = 0.0
            search_entry["current_path_list_r2l"] = [path]
            active_search_list_r2l.append(search_entry)  # batch个 path

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.
            block_AAid_1_l2r = []  # nobi
            block_AAid_2_l2r = []  # nobi
            block_candidate_intensity_l2r = []
            # data blocks to record the current status of search entries
            block_AAid_list_l2r = []
            block_prefix_mass_l2r = []
            block_score_list_l2r = []
            block_score_sum_l2r = []
            block_knapsack_candidate_l2r = []

            block_AAid_1_r2l = []  # nobi
            block_AAid_2_r2l = []  # nobi
            block_candidate_intensity_r2l = []
            # data blocks to record the current status of search entries
            block_AAid_list_r2l = []
            block_prefix_mass_r2l = []
            block_score_list_r2l = []
            block_score_sum_r2l = []
            block_knapsack_candidate_r2l = []

            # store the number of paths of each search entry in the big blocks
            #   to retrieve the info of each search entry later in STEP 4.
            search_entry_size_l2r = [0] * len(active_search_list_l2r)
            search_entry_size_r2l = [0] * len(active_search_list_r2l)

            for entry_index, search_entry in enumerate(active_search_list_l2r):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list_l2r = search_entry["current_path_list_l2r"]
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name_l2r]  # (5, 150000)
                peak_mass_tolerance = peak_batch[0][spectrum_id]["mass_tolerance"]
                for path in current_path_list_l2r:
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        AAid_1 = [AAid_2]

                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]

                    if AAid_2 == LAST_LABEL_l2r:
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch_l2r[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )

                    # get CANDIDATE INTENSITY to predict next AA
                    direction_id = 0
                    # (26,5,40,10)
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    # use knapsack and SUFFIX MASS to filter next AA candidate
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL_l2r]  # 获得后缀质量
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    if not knapsack_candidate and AAid_2 != LAST_LABEL_l2r and AAid_2 != PAD_LABEL:
                        knapsack_candidate.append(LAST_LABEL_l2r)
                    if not knapsack_candidate and (AAid_2 == LAST_LABEL_l2r or AAid_2 == PAD_LABEL):
                        knapsack_candidate.append(PAD_LABEL)
                    block_AAid_1_l2r.append(AAid_1)
                    block_AAid_2_l2r.append(AAid_2)
                    block_candidate_intensity_l2r.append(candidate_intensity)
                    block_AAid_list_l2r.append(AAid_list)
                    block_prefix_mass_l2r.append(prefix_mass)
                    block_score_list_l2r.append(score_list)
                    block_score_sum_l2r.append(score_sum)
                    block_knapsack_candidate_l2r.append(knapsack_candidate)

                    # record the size of each search entry in the blocks
                    search_entry_size_l2r[entry_index] += 1
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                spectrum_id = search_entry["spectrum_id"]
                current_path_list_r2l = search_entry["current_path_list_r2l"]
                precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
                spectrum_original = spectrum_batch[spectrum_id][spectrum_original_name_r2l]  # (5, 150000)
                peak_mass_tolerance = peak_batch[1][spectrum_id]["mass_tolerance"]
                for path in current_path_list_r2l:
                    AAid_list = path["AAid_list"]
                    AAid_2 = AAid_list[-1]
                    if len(AAid_list) > 1:
                        AAid_1 = AAid_list[: len(AAid_list) - 1]
                    else:
                        AAid_1 = [AAid_2]

                    # the current status of this path
                    prefix_mass = path["prefix_mass"]
                    score_list = path["score_list"]
                    score_sum = path["score_sum"]
                    if AAid_2 == LAST_LABEL_r2l:
                        if abs(prefix_mass - precursor_mass) <= peak_mass_tolerance:
                            top_path_batch_r2l[spectrum_id].append(
                                {
                                    "AAid_list": AAid_list,
                                    "score_list": score_list,
                                    "score_sum": score_sum,
                                }
                            )

                    # get CANDIDATE INTENSITY to predict next AA
                    direction_id = 1
                    candidate_intensity = self.get_candidate_intensity(
                        spectrum_original, precursor_mass, prefix_mass, direction_id
                    )
                    # use knapsack and SUFFIX MASS to filter next AA candidate
                    suffix_mass = precursor_mass - prefix_mass - self.mass_ID[LAST_LABEL_r2l]  # 获得后缀质量
                    knapsack_tolerance = int(round(peak_mass_tolerance * self.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidate = self._search_knapsack(suffix_mass, knapsack_tolerance)
                    # if not possible to extend, add LAST_LABEL to end the sequence
                    if not knapsack_candidate and AAid_2 != LAST_LABEL_r2l and AAid_2 != PAD_LABEL:
                        knapsack_candidate.append(LAST_LABEL_r2l)

                    if not knapsack_candidate and (AAid_2 == LAST_LABEL_r2l or AAid_2 == PAD_LABEL):
                        knapsack_candidate.append(PAD_LABEL)

                    # gather data blocks
                    block_AAid_1_r2l.append(AAid_1)
                    block_AAid_2_r2l.append(AAid_2)
                    block_candidate_intensity_r2l.append(candidate_intensity)
                    block_AAid_list_r2l.append(AAid_list)
                    block_prefix_mass_r2l.append(prefix_mass)
                    block_score_list_r2l.append(score_list)
                    block_score_sum_r2l.append(score_sum)
                    block_knapsack_candidate_r2l.append(knapsack_candidate)

                    # record the size of each search entry in the blocks
                    search_entry_size_r2l[entry_index] += 1
            # STEP 3: run tensorflow model on data blocks to predict next AA.
            flag_l2r = [LAST_LABEL_l2r in i for i in block_AAid_list_l2r]
            flag_r2l = [LAST_LABEL_r2l in i for i in block_AAid_list_r2l]
            if not all(flag_l2r) or not all(flag_r2l):
                block_AAid_1_l2r = np.array(block_AAid_1_l2r)  # nobi
                block_AAid_1_l2r = torch.Tensor(block_AAid_1_l2r.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2_l2r = torch.Tensor(np.array(block_AAid_2_l2r)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity_l2r = torch.Tensor(np.array(block_candidate_intensity_l2r)).cuda()
                block_AAid_1_r2l = np.array(block_AAid_1_r2l)  # nobi
                block_AAid_1_r2l = torch.Tensor(block_AAid_1_r2l.transpose(1, 0)).to(torch.int64).cuda()
                block_AAid_2_r2l = torch.Tensor(np.array(block_AAid_2_r2l)).unsqueeze_(0).to(torch.int64).cuda()  # nobi
                block_candidate_intensity_r2l = torch.Tensor(np.array(block_candidate_intensity_r2l)).cuda()
                if block_AAid_1_l2r.size(0) == 1 and block_AAid_1_l2r[0][0] == block_AAid_2_l2r[0][0]:
                    block_decoder_inputs_l2r = block_AAid_1_l2r
                else:
                    block_decoder_inputs_l2r = torch.cat([block_AAid_1_l2r, block_AAid_2_l2r], dim=0)

                if block_AAid_1_r2l.size(0) == 1 and block_AAid_1_r2l[0][0] == block_AAid_2_r2l[0][0]:
                    block_decoder_inputs_r2l = block_AAid_1_r2l
                else:
                    block_decoder_inputs_r2l = torch.cat([block_AAid_1_r2l, block_AAid_2_r2l], dim=0)
                with torch.no_grad():
                    candidate_size_l2r = block_decoder_inputs_l2r.size(1)
                    candidate_size_r2l = block_decoder_inputs_r2l.size(1)
                    current_log_prob_l2r = []
                    current_log_prob_r2l = []
                    for i in range(min(candidate_size_l2r, candidate_size_r2l)):
                        decoder_inputs_l2r = block_decoder_inputs_l2r[:, i].unsqueeze_(1)  # (step - 1, 1)
                        decoder_inputs_r2l = block_decoder_inputs_r2l[:, i].unsqueeze_(1)  # (step - 1, 1)
                        candidate_intensity_l2r = block_candidate_intensity_l2r[i].unsqueeze_(0)
                        candidate_intensity_r2l = block_candidate_intensity_r2l[i].unsqueeze_(0)
                        log_prob_l2r, log_prob_r2l = model.Inference_SB(
                            spectrum_cnn_outputs,
                            candidate_intensity_l2r,
                            candidate_intensity_r2l,
                            decoder_inputs_l2r,
                            decoder_inputs_r2l,
                        )
                        Logsoftmax = torch.nn.LogSoftmax(dim=1)
                        log_prob_l2r = Logsoftmax(log_prob_l2r)
                        current_log_prob_l2r.append(log_prob_l2r)
                        log_prob_r2l = Logsoftmax(log_prob_r2l)
                        current_log_prob_r2l.append(log_prob_r2l)
                    current_log_prob_l2r = torch.cat(current_log_prob_l2r, dim=0)
                    current_log_prob_r2l = torch.cat(current_log_prob_r2l, dim=0)
            if all(flag_l2r) and all(flag_r2l):
                break
            # STEP 4: retrieve data from blocks to update the active_search_list
            #   with knapsack dynamic programming and beam search.
            block_index_l2r = 0
            for entry_index, search_entry in enumerate(active_search_list_l2r):
                # find all possible new paths within knapsack filter
                new_path_list = []
                for index in xrange(
                    block_index_l2r,
                    block_index_l2r + search_entry_size_l2r[entry_index],
                ):
                    for AAid in block_knapsack_candidate_l2r[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list_l2r[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass_l2r[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            new_path["score_list"] = block_score_list_l2r[index] + [current_log_prob_l2r[index][AAid]]
                            new_path["score_sum"] = block_score_sum_l2r[index] + current_log_prob_l2r[index][AAid]
                        else:
                            new_path["score_list"] = block_score_list_l2r[index] + [0.0]
                            new_path["score_sum"] = block_score_sum_l2r[index] + 0.0
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size // 2:
                    new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                    top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]
                    search_entry["current_path_list_l2r"] = [
                        new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                    ]
                else:
                    search_entry["current_path_list_l2r"] = new_path_list
                    search_entry["current_path_list_l2r"].sort(key=lambda t: t["score_sum"], reverse=True)
                block_index_l2r += search_entry_size_l2r[entry_index]
            active_search_list_l2r = [x for x in active_search_list_l2r if x["current_path_list_l2r"]]
            block_index_r2l = 0
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                new_path_list = []
                for index in xrange(
                    block_index_r2l,
                    block_index_r2l + search_entry_size_r2l[entry_index],
                ):
                    for AAid in block_knapsack_candidate_r2l[index]:
                        new_path = {}
                        new_path["AAid_list"] = block_AAid_list_r2l[index] + [AAid]
                        new_path["prefix_mass"] = block_prefix_mass_r2l[index] + self.mass_ID[AAid]
                        if AAid > 2:  # do NOT add score of GO, EOS, PAD
                            new_path["score_list"] = block_score_list_r2l[index] + [current_log_prob_r2l[index][AAid]]
                            new_path["score_sum"] = block_score_sum_r2l[index] + current_log_prob_r2l[index][AAid]
                        else:
                            new_path["score_list"] = block_score_list_r2l[index] + [0.0]
                            new_path["score_sum"] = block_score_sum_r2l[index] + 0.0
                        new_path_list.append(new_path)
                # beam search to select top candidates
                if len(new_path_list) > self.beam_size // 2:
                    new_path_score = np.array([x["score_sum"].cpu() for x in new_path_list])
                    top_k_index = np.argsort(-new_path_score)[: self.beam_size // 2]
                    search_entry["current_path_list_r2l"] = [
                        new_path_list[top_k_index[x]] for x in xrange(self.beam_size // 2)
                    ]
                else:
                    search_entry["current_path_list_r2l"] = new_path_list
                    search_entry["current_path_list_r2l"].sort(key=lambda t: t["score_sum"], reverse=True)
                block_index_r2l += search_entry_size_r2l[entry_index]
            active_search_list_r2l = [x for x in active_search_list_r2l if x["current_path_list_r2l"]]
        return top_path_batch_l2r, top_path_batch_r2l

    def _search_denovo_batch(self, spectrum_batch, model):
        spectrum_batch_size = len(spectrum_batch)
        peak_list = self._select_peak(spectrum_batch)
        top_candidate_batch = [[] for x in xrange(spectrum_batch_size)]
        for peak_batch in peak_list:
            forward_path_batch = self._extend_peak_batch("forward", model, spectrum_batch, peak_batch)
            backward_path_batch = self._extend_peak_batch("backward", model, spectrum_batch, peak_batch)
            for spectrum_id in xrange(spectrum_batch_size):
                if (not forward_path_batch[spectrum_id]) or (not backward_path_batch[spectrum_id]):  # any list is empty
                    continue
                else:
                    for x_path in forward_path_batch[spectrum_id]:
                        for y_path in backward_path_batch[spectrum_id]:
                            AAid_list_forward = x_path["AAid_list"][1:-1]
                            score_list_forward = x_path["score_list"][1:-1]
                            score_sum_forward = x_path["score_sum"]
                            AAid_list_backward = y_path["AAid_list"][1:-1]
                            score_list_backward = y_path["score_list"][1:-1]
                            score_sum_backward = y_path["score_sum"]
                            # reverse backward lists
                            AAid_list_backward = AAid_list_backward[::-1]
                            score_list_backward = score_list_backward[::-1]
                            sequence = AAid_list_backward + AAid_list_forward
                            position_score = score_list_backward + score_list_forward
                            score = score_sum_backward + score_sum_forward
                            top_candidate_batch[spectrum_id].append(
                                {
                                    "sequence": sequence,
                                    "position_score": position_score,
                                    "score": score,
                                }
                            )
        # refine and select the best sequence for each spectrum
        predicted_batch = self._select_sequence(spectrum_batch, top_candidate_batch)

        return predicted_batch

    # test no concat
    def _search_denovo_batch_bi_SB_1(self, spectrum_batch, model):
        spectrum_batch_size = len(spectrum_batch)
        peak_list = self._select_peak(spectrum_batch)
        top_candidate_batch = [[] for x in xrange(spectrum_batch_size)]
        forward_path_batch, backward_path_batch = self._extend_peak_SB_batch(model, spectrum_batch, peak_list)
        for spectrum_id in xrange(spectrum_batch_size):
            if (not forward_path_batch[spectrum_id]) or (not backward_path_batch[spectrum_id]):  # any list is empty
                continue
            else:
                for x_path in forward_path_batch[spectrum_id]:
                    AAid_list_forward = x_path["AAid_list"][1:-1]
                    score_list_forward = x_path["score_list"][1:-1]
                    score_sum_forward = x_path["score_sum"]
                    top_candidate_batch[spectrum_id].append(
                        {
                            "sequence": AAid_list_forward,
                            "position_score": score_list_forward,
                            "score": score_sum_forward,
                        }
                    )
                for y_path in backward_path_batch[spectrum_id]:
                    AAid_list_backward = y_path["AAid_list"][1:-1]
                    score_list_backward = y_path["score_list"][1:-1]
                    score_sum_backward = y_path["score_sum"]
                    # reverse backward lists
                    AAid_list_backward = AAid_list_backward[::-1]
                    score_list_backward = score_list_backward[::-1]
                    top_candidate_batch[spectrum_id].append(
                        {
                            "sequence": AAid_list_backward,
                            "position_score": score_list_backward,
                            "score": score_sum_backward,
                        }
                    )
        # refine and select the best sequence for each spectrum
        predicted_batch = self._select_sequence(spectrum_batch, top_candidate_batch)
        return predicted_batch

    def _search_denovo_batch_bi_indepedent(self, spectrum_batch, model):
        spectrum_batch_size = len(spectrum_batch)
        peak_list = self._select_peak(spectrum_batch)
        top_candidate_batch_forward = [[] for x in xrange(spectrum_batch_size)]
        top_candidate_batch_backward = [[] for x in xrange(spectrum_batch_size)]
        for peak_batch in peak_list:
            forward_path_batch = self._extend_peak_batch("forward", model, spectrum_batch, peak_batch)
            backward_path_batch = self._extend_peak_batch("backward", model, spectrum_batch, peak_batch)
            # concatenate forward and backward paths
            for spectrum_id in xrange(spectrum_batch_size):
                if (not forward_path_batch[spectrum_id]) or (not backward_path_batch[spectrum_id]):  # any list is empty
                    continue
                else:
                    for x_path in forward_path_batch[spectrum_id]:
                        AAid_list_forward = x_path["AAid_list"][1:-1]
                        score_list_forward = x_path["score_list"][1:-1]
                        score_sum_forward = x_path["score_sum"]

                        top_candidate_batch_forward[spectrum_id].append(
                            {
                                "sequence": AAid_list_forward,
                                "position_score": score_list_forward,
                                "score": score_sum_forward,
                            }
                        )
                    for y_path in backward_path_batch[spectrum_id]:
                        AAid_list_backward = y_path["AAid_list"][1:-1]
                        score_list_backward = y_path["score_list"][1:-1]
                        score_sum_backward = y_path["score_sum"]
                        # reverse backward lists
                        AAid_list_backward = AAid_list_backward[::-1]
                        score_list_backward = score_list_backward[::-1]
                        top_candidate_batch_backward[spectrum_id].append(
                            {
                                "sequence": AAid_list_backward,
                                "position_score": score_list_backward,
                                "score": score_sum_backward,
                            }
                        )
        # refine and select the best sequence for each spectrum
        predicted_batch_forward = self._select_sequence(spectrum_batch, top_candidate_batch_forward)
        predicted_batch_backward = self._select_sequence(spectrum_batch, top_candidate_batch_backward)
        return predicted_batch_forward, predicted_batch_backward

    def _search_denovo_batch_bi_SB(self, spectrum_batch, model):
        spectrum_batch_size = len(spectrum_batch)
        peak_list = self._select_peak(spectrum_batch)
        top_candidate_batch_forward = [[] for x in xrange(spectrum_batch_size)]
        top_candidate_batch_backward = [[] for x in xrange(spectrum_batch_size)]
        forward_path_batch, backward_path_batch = self._extend_peak_SB_batch(model, spectrum_batch, peak_list)
        for spectrum_id in xrange(spectrum_batch_size):
            if (not forward_path_batch[spectrum_id]) or (not backward_path_batch[spectrum_id]):  # any list is empty
                continue
            else:
                for x_path in forward_path_batch[spectrum_id]:
                    AAid_list_forward = x_path["AAid_list"][1:-1]
                    score_list_forward = x_path["score_list"][1:-1]
                    score_sum_forward = x_path["score_sum"]

                    top_candidate_batch_forward[spectrum_id].append(
                        {
                            "sequence": AAid_list_forward,
                            "position_score": score_list_forward,
                            "score": score_sum_forward,
                        }
                    )
                for y_path in backward_path_batch[spectrum_id]:
                    AAid_list_backward = y_path["AAid_list"][1:-1]
                    score_list_backward = y_path["score_list"][1:-1]
                    score_sum_backward = y_path["score_sum"]
                    # reverse backward lists
                    AAid_list_backward = AAid_list_backward[::-1]
                    score_list_backward = score_list_backward[::-1]
                    top_candidate_batch_backward[spectrum_id].append(
                        {
                            "sequence": AAid_list_backward,
                            "position_score": score_list_backward,
                            "score": score_sum_backward,
                        }
                    )
        # refine and select the best sequence for each spectrum
        predicted_batch_forward = self._select_sequence(spectrum_batch, top_candidate_batch_forward)
        predicted_batch_backward = self._select_sequence(spectrum_batch, top_candidate_batch_backward)
        return predicted_batch_forward, predicted_batch_backward

    def _search_knapsack(self, mass, knapsack_tolerance):
        # convert the mass and tolerance to a range of columns of knapsack_matrix
        mass_round = int(round(mass * self.KNAPSACK_AA_RESOLUTION))  #
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        if mass_upperbound < self.mass_AA_min_round:  # 57.0215
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_AAid = np.flatnonzero(
            np.any(
                self.knapsack_matrix[:, mass_lowerbound_col : mass_upperbound_col + 1],
                # pylint: disable=line-too-long
                axis=1,
            )
        )
        return candidate_AAid.tolist()

    def _select_peak(self, spectrum_batch):
        peak_list = []
        spectrum_batch_size = len(spectrum_batch)
        mass_GO = self.mass_ID[self.GO_ID]
        peak_batch = [
            {
                "prefix_mass": mass_GO,
                "suffix_mass": x["precursor_mass"] - mass_GO,
                "mass_tolerance": self.precursor_mass_tolerance,
            }
            for x in spectrum_batch
        ]  # list of dict, size= batchsize
        peak_list.append(peak_batch)
        mass_EOS = self.mass_ID[self.EOS_ID]
        peak_batch = [
            {
                "prefix_mass": x["precursor_mass"] - mass_EOS,
                "suffix_mass": mass_EOS,
                "mass_tolerance": self.precursor_mass_tolerance,
            }
            for x in spectrum_batch
        ]  # list of dict, size= batchsize
        peak_list.append(peak_batch)
        argmax_mass_batch = []
        argmax_mass_complement_batch = []
        for spectrum in spectrum_batch:
            precursor_mass = spectrum["precursor_mass"]
            precursor_mass_C = precursor_mass - mass_EOS
            precursor_mass_C_location = int(round(precursor_mass_C * self.SPECTRUM_RESOLUTION))
            spectrum_forward = spectrum["spectrum_original_forward"]  # (5, 150000)
            argmax_location = np.argpartition(
                -spectrum_forward[self.neighbor_center, :precursor_mass_C_location],
                self.num_position,
            )[
                : self.num_position
            ]  # pylint: disable=line-too-long
            # NOTE that the precursor mass tolerance from now on should depend on
            argmax_mass = argmax_location / self.SPECTRUM_RESOLUTION
            argmax_mass_complement = [(precursor_mass - x) for x in argmax_mass]
            argmax_mass_batch.append(argmax_mass)
            argmax_mass_complement_batch.append(argmax_mass_complement)

        # NOTE that the peak mass tolerance now depends on SPECTRUM_RESOLUTION,
        #   because the peak was selected from the ms2 spectrum
        mass_tolerance = 1.0 / self.SPECTRUM_RESOLUTION
        for index in xrange(self.num_position):
            peak_batch = [
                {
                    "prefix_mass": b[index],
                    "suffix_mass": y[index],
                    "mass_tolerance": mass_tolerance,
                }
                for b, y in zip(argmax_mass_batch, argmax_mass_complement_batch)
            ]
            peak_list.append(peak_batch)
            peak_batch = [
                {
                    "prefix_mass": b[index],
                    "suffix_mass": y[index],
                    "mass_tolerance": mass_tolerance,
                }
                for b, y in zip(argmax_mass_complement_batch, argmax_mass_batch)
            ]
            peak_list.append(peak_batch)

        return peak_list

    def _select_sequence(self, spectrum_batch, top_candidate_batch):
        spectrum_batch_size = len(spectrum_batch)
        # refine/filter predicted sequences by precursor mass,
        #   especially for middle peak extension
        refine_batch = [[] for x in xrange(spectrum_batch_size)]
        for spectrum_id in xrange(spectrum_batch_size):
            precursor_mass = spectrum_batch[spectrum_id]["precursor_mass"]
            candidate_list = top_candidate_batch[spectrum_id]
            for candidate in candidate_list:
                sequence = candidate["sequence"]
                sequence_mass = sum(self.mass_ID[x] for x in sequence)
                sequence_mass += self.mass_ID[self.GO_ID] + self.mass_ID[self.EOS_ID]
                if abs(sequence_mass - precursor_mass) <= self.precursor_mass_tolerance:
                    refine_batch[spectrum_id].append(candidate)
        # select the best len-normalized scoring candidate
        predicted_batch = [[] for x in xrange(spectrum_batch_size)]
        for spectrum_id in xrange(spectrum_batch_size):
            predicted_batch[spectrum_id] = {}
            predicted_batch[spectrum_id]["feature_id"] = spectrum_batch[spectrum_id]["feature_id"]
            predicted_batch[spectrum_id]["feature_area"] = spectrum_batch[spectrum_id]["feature_area"]
            predicted_batch[spectrum_id]["precursor_mz"] = spectrum_batch[spectrum_id]["precursor_mz"]
            predicted_batch[spectrum_id]["precursor_charge"] = spectrum_batch[spectrum_id]["precursor_charge"]
            predicted_batch[spectrum_id]["scan_list_middle"] = spectrum_batch[spectrum_id]["scan_list_middle"]
            predicted_batch[spectrum_id]["scan_list_original"] = spectrum_batch[spectrum_id]["scan_list_original"]
            candidate_list = refine_batch[spectrum_id]
            if not candidate_list:  # cannot find any peptide
                predicted_batch[spectrum_id]["sequence"] = [[]]
                predicted_batch[spectrum_id]["position_score"] = [[]]
                predicted_batch[spectrum_id]["score"] = [[]]
            else:
                score_array = np.array([x["score"].cpu() / len(x["sequence"]) for x in candidate_list])
                if len(candidate_list) > self.topk_output:
                    topk_index = np.argpartition(-score_array, self.topk_output)[
                        : self.topk_output
                    ]  # pylint: disable=line-too-long
                    predicted_list = [candidate_list[index] for index in topk_index]
                else:
                    predicted_list = candidate_list
                predicted_batch[spectrum_id]["score"] = [
                    predicted["score"].cpu() / len(predicted["sequence"]) for predicted in predicted_list
                ]
                predicted_batch[spectrum_id]["position_score"] = [
                    predicted["position_score"] for predicted in predicted_list
                ]
                # NOTE that we convert AAid back to letter
                predicted_batch[spectrum_id]["sequence"] = [
                    [self.vocab_reverse[x] for x in predicted["sequence"]] for predicted in predicted_list
                ]
        return predicted_batch
