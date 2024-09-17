
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import List
import numpy as np
import torch

from Biatnovo.test_true_feeding import test_logit_batch_2_score
from model import InferenceModelWrapper, device
from v2 import deepnovo_config
from v2.data_reader import BatchDenovoData, DIAFeature, DenovoData, TrainData, collate_func
from DataProcess.deepnovo_cython_modules import get_candidate_intensity
from v2.writer import BeamSearchedSequence, DenovoWriter
logger = logging.getLogger(__name__)


class Direction(Enum):
    forward = 1
    backward = 2


@dataclass
class BeamSearchStartPoint:
    prefix_mass: float
    suffix_mass: float
    mass_tolerance: float
    direction: Direction

@dataclass
class SearchEntry:
    feature_index: int
    current_path_list: list  # list of search paths

@dataclass
class SearchPath:
    aa_id_list: list
    aa_seq_mass: float
    score_list: list
    score_sum: float
    direction: Direction

@dataclass
class DenovoResult:
    dia_feature: DIAFeature
    best_beam_search_sequence: BeamSearchedSequence

class KnapsackSearcher(object):
    def __init__(self, MZ_MAX, knapsack_file):
        self.knapsack_file = knapsack_file
        self.MZ_MAX = MZ_MAX
        self.knapsack_aa_resolution = deepnovo_config.KNAPSACK_AA_RESOLUTION
        if os.path.isfile(knapsack_file):
            logging.info("KnapsackSearcher.__init__(): load knapsack matrix")
            self.knapsack_matrix = np.load(knapsack_file)
        else:
            logging.info("KnapsackSearcher.__init__(): build knapsack matrix from scratch")
            self.knapsack_matrix = self._build_knapsack()

    def _build_knapsack(self):
        max_mass = self.MZ_MAX - deepnovo_config.mass_N_terminus - deepnovo_config.mass_C_terminus
        max_mass_round = int(round(max_mass * self.knapsack_aa_resolution))
        max_mass_upperbound = max_mass_round + self.knapsack_aa_resolution
        knapsack_matrix = np.zeros(shape=(deepnovo_config.vocab_size, max_mass_upperbound), dtype=bool)
        for aa_id in range(3, deepnovo_config.vocab_size):
            mass_aa = int(round(deepnovo_config.mass_ID[aa_id] * self.knapsack_aa_resolution))

            for col in range(max_mass_upperbound):
                current_mass = col + 1
                if current_mass < mass_aa:
                    knapsack_matrix[aa_id, col] = False
                elif current_mass == mass_aa:
                    knapsack_matrix[aa_id, col] = True
                elif current_mass > mass_aa:
                    sub_mass = current_mass - mass_aa
                    sub_col = sub_mass - 1
                    if np.sum(knapsack_matrix[:, sub_col]) > 0:
                        knapsack_matrix[aa_id, col] = True
                        knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col], knapsack_matrix[:, sub_col])
                    else:
                        knapsack_matrix[aa_id, col] = False
        np.save(self.knapsack_file, knapsack_matrix)
        return knapsack_matrix

    def search_knapsack(self, mass, knapsack_tolerance):
        mass_round = int(round(mass * self.knapsack_aa_resolution))
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        if mass_upperbound < deepnovo_config.mass_AA_min_round:
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)],
                                                axis=1))
        return candidate_aa_id.tolist()

class DeepNovoAttionDenovo():
    def __init__(self, MZ_MAX, knapsack_file, beam_size) -> None:
        self.MZ_MAX = MZ_MAX  # legacy, not used here
        self.beam_size = beam_size
        self.knapsack_searcher = KnapsackSearcher(MZ_MAX, knapsack_file)

    @staticmethod
    def _get_start_point(batch_denovo_data: BatchDenovoData) -> tuple:
        mass_GO = deepnovo_config.mass_ID[deepnovo_config.GO_ID]
        forward_start_point_lists = [BeamSearchStartPoint(prefix_mass=mass_GO,
                                                          suffix_mass=dia_feature.precursor_mass - mass_GO,
                                                          mass_tolerance=deepnovo_config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                          direction=Direction.forward)
                                     for dia_feature in batch_denovo_data.dia_features]

        mass_EOS = deepnovo_config.mass_ID[deepnovo_config.EOS_ID]
        backward_start_point_lists = [BeamSearchStartPoint(prefix_mass=dia_features.precursor_mass - mass_EOS,
                                                           suffix_mass=mass_EOS,
                                                           mass_tolerance=deepnovo_config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                           direction=Direction.backward)
                                      for dia_features in batch_denovo_data.dia_features]
        return forward_start_point_lists, backward_start_point_lists

    @staticmethod
    def _select_path(batch_denovo_data: BatchDenovoData, top_candidate_batch: list) -> list:
        """
        for each feature, select the best denovo sequence given by DeepNovo model
        :param batch_denovo_data:
        :param top_candidate_batch: defined in _search_denovo_batch
        :return:
        list of DenovoResult
        """
        feature_batch_size = len(batch_denovo_data.dia_features)

        refine_batch = [[] for x in range(feature_batch_size)]
        for feature_index in range(feature_batch_size):
            precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
            candidate_list = top_candidate_batch[feature_index]
            for beam_search_sequence in candidate_list:
                sequence = beam_search_sequence.sequence
                sequence_mass = sum(deepnovo_config.mass_ID[x] for x in sequence)
                sequence_mass += deepnovo_config.mass_ID[deepnovo_config.GO_ID] + deepnovo_config.mass_ID[
                    deepnovo_config.EOS_ID]
                if abs(sequence_mass - precursor_mass) <= deepnovo_config.PRECURSOR_MASS_PRECISION_TOLERANCE:
                    #logger.debug(f"sequence {sequence} of feature "
                    #             f"{batch_denovo_data.dia_features[feature_index].feature_id} refined")
                    refine_batch[feature_index].append(beam_search_sequence)
        predicted_batch = []
        for feature_index in range(feature_batch_size):
            candidate_list = refine_batch[feature_index]
            if not candidate_list:
                best_beam_search_sequence = BeamSearchedSequence(
                    sequence=[],
                    position_score=[],
                    score=-float('inf')
                )
            else:
                # sort candidate sequence by average position score
                best_beam_search_sequence = max(candidate_list, key=lambda x: x.score)

            denovo_result = DenovoResult(
                dia_feature=batch_denovo_data.dia_features[feature_index],
                best_beam_search_sequence=best_beam_search_sequence
            )
            predicted_batch.append(denovo_result)
        return predicted_batch


    def _beam_search(self, model_wrapper : InferenceModelWrapper, batch_denovo_data: BatchDenovoData, start_point_batch: list):
        num_features = len(batch_denovo_data.dia_features)
        top_path_batch = [[] for _ in range(num_features)]

        direction_cint_map = {Direction.forward: 0, Direction.backward: 1}
        direction = start_point_batch[0].direction
        if direction == Direction.forward:
            get_start_mass = lambda x: x.prefix_mass
            first_label = deepnovo_config.GO_ID
            last_label = deepnovo_config.EOS_ID
        elif direction == Direction.backward:
            get_start_mass = lambda x: x.suffix_mass
            first_label = deepnovo_config.EOS_ID
            last_label = deepnovo_config.GO_ID
        else:
            raise ValueError('direction neither forward nor backward')

        # step 1: extract original spectrum
        spectrum_cnn_outputs = model_wrapper.init_spectrum_cnn(batch_denovo_data.spectrum_holder) # (batchszie, 16, 256)

        # initialize activate search list
        active_search_list = []
        for feature_index in range(num_features):  # batchsize
            # all feature in the same batch should be from same direction
            assert direction == start_point_batch[feature_index].direction
            path = SearchPath(
                aa_id_list=[first_label],
                aa_seq_mass=get_start_mass(start_point_batch[feature_index]),
                score_list=[0.0],
                score_sum=0.0,
                direction=direction,
            )
            search_entry = SearchEntry(
                feature_index=feature_index,
                current_path_list=[path]
            )
            active_search_list.append(search_entry)

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        current_step = 0
        while True:
            # STEP 2: gather data from active search entries and group into blocks.

            # model input
            block_intensity_input = []
            # data stored in path
            block_aa_id_list = []
            block_aa_seq_mass = []
            block_score_list = []
            block_score_sum = []
            block_spectrum_cnn_outputs = []
            block_knapsack_candidates = []
            # store the number of paths of each search entry in the big blocks
            #     to retrieve the info of each search entry later in STEP 4.
            search_entry_size = [0] * len(active_search_list)

            for entry_index, search_entry in enumerate(active_search_list):
                # 迭代batchsize
                feature_index = search_entry.feature_index
                current_path_list = search_entry.current_path_list
                precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
                peak_mass_tolerance = start_point_batch[feature_index].mass_tolerance
                if direction == Direction.forward:
                    spectrum_original = batch_denovo_data.spectrum_original_forward[feature_index]
                elif direction == Direction.backward:
                    spectrum_original = batch_denovo_data.spectrum_original_backward[feature_index]

                for path in current_path_list: # eg: SearchPath(aaid_id_list=[1, 9])
                    aa_id_list = path.aa_id_list
                    aa_id = aa_id_list[-1]
                    score_sum = path.score_sum
                    aa_seq_mass = path.aa_seq_mass
                    score_list = path.score_list

                    if aa_seq_mass > precursor_mass + peak_mass_tolerance:
                        # 终止条件：aa_seq_mass超过了precursor_mass+peak_mass_tolerance
                        continue

                    if aa_id == last_label:
                        # 终止条件：最后一个aa是EOS
                        if abs(aa_seq_mass - precursor_mass) <= peak_mass_tolerance:
                            seq = aa_id_list[1:-1]
                            trunc_score_list = score_list[1:-1]
                            if direction == Direction.backward:
                                seq = seq[::-1]
                                trunc_score_list = trunc_score_list[::-1]

                            top_path_batch[feature_index].append(
                                BeamSearchedSequence(sequence=seq,
                                                     position_score=trunc_score_list,
                                                     score=path.score_sum / len(seq))
                            )
                        continue

                    candidate_intensity = get_candidate_intensity(spectrum_original,
                                                                  precursor_mass,
                                                                  aa_seq_mass,
                                                                  direction_cint_map[direction])
                    residual_mass = precursor_mass - aa_seq_mass - deepnovo_config.mass_ID[last_label]
                    knapsack_tolerance = int(round(peak_mass_tolerance * deepnovo_config.KNAPSACK_AA_RESOLUTION))
                    # list of avaliable aa_id
                    knapsack_candidates = self.knapsack_searcher.search_knapsack(residual_mass, knapsack_tolerance)

                    if not knapsack_candidates:
                        # if not possible aa, force it to stop.
                        knapsack_candidates.append(last_label)

                    block_intensity_input.append(candidate_intensity)
                    block_aa_id_list.append(aa_id_list)
                    block_aa_seq_mass.append(aa_seq_mass)
                    block_score_list.append(score_list)
                    block_score_sum.append(score_sum)
                    block_knapsack_candidates.append(knapsack_candidates)
                    block_spectrum_cnn_outputs.append(spectrum_cnn_outputs[entry_index, :, :].unsqueeze_(0))
                    # record the size of each search entry in the blocks
                    search_entry_size[entry_index] += 1

            # step 3 run model on data blocks to predict next AA.
            # output is stored in current_log_prob
            if not block_intensity_input:
                break
            block_intensity_input = torch.from_numpy(np.array(block_intensity_input)).to(device)
            block_decoder_inputs = torch.from_numpy(np.array(block_aa_id_list).transpose(1, 0)).to(device)
            block_spectrum_cnn_outputs = torch.cat(block_spectrum_cnn_outputs, dim=0)

            current_log_prob = model_wrapper.inference(
                block_spectrum_cnn_outputs,
                block_intensity_input, # (batchsize * beamsize, 26)
                block_decoder_inputs,  # (seq_len, batchsize)
                direction_id=direction_cint_map[direction]
            )
            
            Logsoftmax = torch.nn.LogSoftmax(dim=1)
            current_log_prob = Logsoftmax(current_log_prob)
            # transfer log_prob back to cpu
            current_log_prob = current_log_prob.cpu().numpy()

            # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
                        # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list):
                new_path_list = []
                direction = search_entry.current_path_list[0].direction
                for index in range(block_index, block_index + search_entry_size[entry_index]):
                    for aa_id in block_knapsack_candidates[index]:  # < batchsize * beamsize
                        if aa_id > 2:
                            # do not add score of GO, EOS, PAD
                            new_score_list = block_score_list[index] + [current_log_prob[index][aa_id]]
                            new_score_sum = block_score_sum[index] + current_log_prob[index][aa_id]
                        else:
                            new_score_list = block_score_list[index] + [0.0]
                            new_score_sum = block_score_sum[index] + 0.0

                        new_path = SearchPath(
                            aa_id_list=block_aa_id_list[index] + [aa_id],
                            aa_seq_mass=block_aa_seq_mass[index] + deepnovo_config.mass_ID[aa_id],
                            score_list=new_score_list,
                            score_sum=new_score_sum,
                            direction=direction
                        )
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x.score_sum for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size]
                    search_entry.current_path_list = [new_path_list[ii] for ii in top_k_index]  # list of SearchPath
                else:
                    search_entry.current_path_list = new_path_list

                block_index += search_entry_size[entry_index]

            # remove search entry with no path
            active_search_list = [x for x in active_search_list if x.current_path_list]  # batchsize

            current_step += 1

            if not active_search_list:
                break

        return top_path_batch


    def search_denovo(self, model_wrapper: InferenceModelWrapper,
                      beam_search_reader: torch.utils.data.DataLoader,
                      denovo_writer : DenovoWriter) -> List[DenovoResult]:
        logger.info("start beam search denovo")
        predicted_denovo_list = []
        total_batch_num = int(len(beam_search_reader.dataset) / deepnovo_config.batch_size_predict)
        for index, batch_denovo_data in enumerate(beam_search_reader):
            logger.info("Read {}th/{} batches".format(index, total_batch_num))
            if deepnovo_config.is_sb:
                predicted_batch = self._sb_search_denovo_batch(batch_denovo_data, model_wrapper)
                predicted_denovo_list += predicted_batch
                for denovo_result in predicted_batch:
                    denovo_writer.write(denovo_result.dia_feature, denovo_result.best_beam_search_sequence)
            else:
                predicted_forward_batch, predicted_backward_batch = self._search_denovo_batch(batch_denovo_data, model_wrapper)
                for denovo_forward_result, denvo_backward_result in zip(predicted_forward_batch, predicted_backward_batch):
                    denovo_writer.write_sequences(denovo_forward_result.dia_feature, denovo_forward_result.best_beam_search_sequence, denvo_backward_result.best_beam_search_sequence)
                # predicted_batch = self._sb_search_denovo_batch(batch_denovo_data, model_wrapper)
                # predicted_denovo_list += predicted_batch
                # for denovo_result in predicted_batch:
                #     denovo_writer.write(denovo_result.dia_feature, denovo_result.best_beam_search_sequence)

        return predicted_denovo_list

    # 返回的结果包含预测本身
    def concate_more(self, forward_sequence, backward_sequence, precursor_mass):
        forward_sequence = [deepnovo_config.vocab_reverse[aa_id] for
                                           aa_id in forward_sequence]
        backward_sequence = [deepnovo_config.vocab_reverse[aa_id] for aa_id in backward_sequence]
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
                peptide_ids = [deepnovo_config.vocab[x] for x in str_tmp2]
                mass_tmp = sum(deepnovo_config.mass_ID[x] for x in peptide_ids)
                if mass_tmp < end_mass and mass_tmp > begin_mass:
                    result.append(str_tmp2)
        result.append(forward_sequence)
        result.append(backward_sequence)
        result = list(set([tuple(t) for t in result]))
        return result


    # def _search_denovo_batch(self, batch_denovo_data: BatchDenovoData, model_wrapper: InferenceModelWrapper) -> List[DenovoResult]:
    #     start_time = time.time()
    #     feature_batch_size = len(batch_denovo_data.dia_features)
    #     start_points_tuple = self._get_start_point(batch_denovo_data)
    #     top_candidate_batch = [[] for x in range(feature_batch_size)]

    #     for start_points in start_points_tuple:
    #         beam_search_result_batch = self._beam_search(model_wrapper, batch_denovo_data, start_points)
    #         for feature_index in range(feature_batch_size):
    #             top_candidate_batch[feature_index].extend(beam_search_result_batch[feature_index])
    #     predicted_batch = self._select_path(batch_denovo_data, top_candidate_batch)
    #     test_time = time.time() - start_time
    #     logger.info("beam_search(): batch time {}s".format(test_time))
    #     return predicted_batch
    
    def _sb_search_denovo_batch(self, batch_denovo_data: BatchDenovoData, model_wrapper: InferenceModelWrapper) -> List[DenovoResult]:
        start_time = time.time()
        feature_batch_size = len(batch_denovo_data.dia_features)
        forward_start_point_lists, backward_start_point_lists = self._get_start_point(batch_denovo_data)
        top_candidate_batch = [[] for x in range(feature_batch_size)]

        forward_beam_search_result_batch, backward_beam_search_result_batch = self._sb_beam_search(model_wrapper, batch_denovo_data, forward_start_point_lists, backward_start_point_lists)
        if deepnovo_config.concat_more:
            top_forward_denovo_result = self._select_path(batch_denovo_data, forward_beam_search_result_batch)
            top_backward_denovo_result = self._select_path(batch_denovo_data, backward_beam_search_result_batch)
            denovo_data_list = self._denovo_decollate_func(batch_denovo_data)
            sepctrum_index = []
            train_data_list = []
            predicted_batch = []
            concate_result = []
            
            for feature_index in range(feature_batch_size):
                forward_sequence = top_forward_denovo_result[feature_index].best_beam_search_sequence.sequence
                backward_sequence = top_backward_denovo_result[feature_index].best_beam_search_sequence.sequence
                precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
                if len(forward_sequence) > 0 and len(backward_sequence) > 0:
                    concate_result = self.concate_more(forward_sequence, backward_sequence, precursor_mass)
                elif len(forward_sequence) > 0:
                    forward_sequence = [deepnovo_config.vocab_reverse[aa_id] for
                                               aa_id in forward_sequence]
                    concate_result.append(forward_sequence)
                else:  #F2:1241
                    backward_sequence = [deepnovo_config.vocab_reverse[aa_id] for aa_id in backward_sequence]
                    concate_result.append(backward_sequence)
                for sequence in concate_result:
                    train_data = self._convert2training_data(denovo_data_list[feature_index], sequence)
                    train_data_list.append(train_data)
                    sepctrum_index.append(feature_index)
                
            bi_score = self.test_accuracy_score_v2(train_data_list, model_wrapper)
            result_score = [[] for x in range(feature_batch_size)]
            index = 0
            for x in bi_score:
                result_score[sepctrum_index[index]].append(x)
                index += 1
        
        
            for feature_index in range(feature_batch_size):
                score_sum_list = []
                for result in result_score[feature_index]:
                    score_sum_list.append(result["score_sum"])
                if len(score_sum_list) > 0:
                    tmp_i = score_sum_list.index(max(score_sum_list))
                    best_beam_search_sequence = BeamSearchedSequence(sequence=result_score[feature_index][tmp_i]["aaid_sequence"],
                                         position_score=result_score[feature_index][tmp_i]["position_score"],
                                         score=score_sum_list[tmp_i])
                    denovo_result = DenovoResult(
                        dia_feature=batch_denovo_data.dia_features[feature_index],
                        best_beam_search_sequence=best_beam_search_sequence
                    )
                    predicted_batch.append(denovo_result)

            test_time = time.time() - start_time
            logger.info("beam_search(): batch time {}s".format(test_time))
            return predicted_batch
        else:
            for feature_index in range(feature_batch_size):
                top_candidate_batch[feature_index].extend(forward_beam_search_result_batch[feature_index])
                top_candidate_batch[feature_index].extend(backward_beam_search_result_batch[feature_index])
            predicted_batch = self._select_path(batch_denovo_data, top_candidate_batch)
            test_time = time.time() - start_time
            logger.info("beam_search(): batch time {}s".format(test_time))
            return predicted_batch

   

    def _search_denovo_batch(self, batch_denovo_data: BatchDenovoData, model_wrapper: InferenceModelWrapper) -> tuple:
        start_time = time.time()
        feature_batch_size = len(batch_denovo_data.dia_features)
        forward_start_point_lists, backward_start_point_lists = self._get_start_point(batch_denovo_data)
        top_candidate_batch = [[] for x in range(feature_batch_size)]

        forward_beam_search_result_batch, backward_beam_search_result_batch = self._sb_beam_search(model_wrapper, batch_denovo_data, forward_start_point_lists, backward_start_point_lists)
        top_forward_denovo_result = self._select_path(batch_denovo_data, forward_beam_search_result_batch)
        top_backward_denovo_result = self._select_path(batch_denovo_data, backward_beam_search_result_batch)
        test_time = time.time() - start_time
        logger.info("beam_search(): batch time {}s".format(test_time))
        return top_forward_denovo_result, top_backward_denovo_result


    # def _search_denovo_batch(self, batch_denovo_data: BatchDenovoData, model_wrapper: InferenceModelWrapper) -> List[DenovoResult]:
    #     start_time = time.time()
    #     feature_batch_size = len(batch_denovo_data.dia_features)
    #     start_points_tuple = self._get_start_point(batch_denovo_data)

    #     top_candidate_forward_batch = self._beam_search(model_wrapper, batch_denovo_data, start_points_tuple[0])
    #     top_forward_denovo_result = self._select_path(batch_denovo_data, top_candidate_forward_batch)
    #     top_candidate_backward_batch = self._beam_search(model_wrapper, batch_denovo_data, start_points_tuple[1])
    #     top_backward_denovo_result = self._select_path(batch_denovo_data, top_candidate_backward_batch)
    #     denovo_data_list = self._denovo_decollate_func(batch_denovo_data)  # tensor to list
    #     sepctrum_index = []
    #     train_data_list = []
    #     predicted_batch = []
    #     concate_result = []
        
    #     for feature_index in range(feature_batch_size):
    #         forward_sequence = top_forward_denovo_result[feature_index].best_beam_search_sequence.sequence
    #         backward_sequence = top_backward_denovo_result[feature_index].best_beam_search_sequence.sequence
    #         precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
    #         if len(forward_sequence) > 0 and len(backward_sequence) > 0:
    #             concate_result = self.concate_more(forward_sequence, backward_sequence, precursor_mass)
    #         elif len(forward_sequence) > 0:
    #             forward_sequence = [deepnovo_config.vocab_reverse[aa_id] for
    #                                        aa_id in forward_sequence]
    #             concate_result.append(forward_sequence)
    #         else:  #F2:1241
    #             backward_sequence = [deepnovo_config.vocab_reverse[aa_id] for aa_id in backward_sequence]
    #             concate_result.append(backward_sequence)
    #         for sequence in concate_result:
    #             train_data = self._convert2training_data(denovo_data_list[feature_index], sequence)
    #             train_data_list.append(train_data)
    #             sepctrum_index.append(feature_index)
            
    #     bi_score = self.test_accuracy_score_v2(train_data_list, model_wrapper)
    #     result_score = [[] for x in range(feature_batch_size)]
    #     index = 0
    #     for x in bi_score:
    #         result_score[sepctrum_index[index]].append(x)
    #         index += 1
    
    #     for feature_index in range(feature_batch_size):
    #         score_sum_list = []
    #         for result in result_score[feature_index]:
    #             score_sum_list.append(result["score_sum"])
    #         if len(score_sum_list) > 0:
    #             tmp_i = score_sum_list.index(max(score_sum_list))
    #             best_beam_search_sequence = BeamSearchedSequence(sequence=result_score[feature_index][tmp_i]["aaid_sequence"],
    #                                  position_score=result_score[feature_index][tmp_i]["position_score"],
    #                                  score=score_sum_list[tmp_i])
    #             denovo_result = DenovoResult(
    #                 dia_feature=batch_denovo_data.dia_features[feature_index],
    #                 best_beam_search_sequence=best_beam_search_sequence
    #             )
    #             predicted_batch.append(denovo_result)

    #     test_time = time.time() - start_time
    #     logger.info("beam_search(): batch time {}s".format(test_time))
    #     return predicted_batch
    
    def test_accuracy_score_v2(self, train_data_list: List[TrainData], model_wrapper: InferenceModelWrapper):
        (
            spectrum_holder,
            batch_intensity_inputs_forward,
            batch_intensity_inputs_backward,
            batch_decoder_inputs_forward,
            batch_decoder_inputs_backward
        ) = collate_func(train_data_list)

        # move to device
        spectrum_holder = spectrum_holder.to(device)
        batch_intensity_inputs_forward = batch_intensity_inputs_forward.to(device)
        batch_intensity_inputs_backward = batch_intensity_inputs_backward.to(device)
        batch_decoder_inputs_forward = batch_decoder_inputs_forward.to(device)
        batch_decoder_inputs_backward = batch_decoder_inputs_backward.to(device)
        # (seq_len, batchsize, 26, 40, 10)
        batch_intensity_inputs_forward = batch_intensity_inputs_forward.permute(1, 0, 2, 3, 4)
        batch_intensity_inputs_backward = batch_intensity_inputs_backward.permute(1, 0, 2, 3, 4)
        # (seq_len, batchsize)
        batch_decoder_inputs_forward = batch_decoder_inputs_forward.permute(1, 0)
        batch_decoder_inputs_backward = batch_decoder_inputs_backward.permute(1, 0)

        spectrum_cnn_outputs = model_wrapper.spectrum_cnn(spectrum_holder).permute(1, 0, 2)  # (batchsize, 1, 256)
        output_logits_forward = model_wrapper.forward_model(spectrum_cnn_outputs, batch_intensity_inputs_forward, batch_decoder_inputs_forward[: -1])
        output_logits_backward = model_wrapper.backward_model(spectrum_cnn_outputs, batch_intensity_inputs_backward, batch_decoder_inputs_backward[: -1])
        output_logits_forward = output_logits_forward.view(-1, output_logits_forward.size(-1))
        output_logits_backward = output_logits_backward.view(-1, output_logits_backward.size(-1))
       
        output_batch = test_logit_batch_2_score(
            batch_decoder_inputs_forward, # (seq_len, batchsize)
            batch_decoder_inputs_backward,
            output_logits_forward, # (seq_len - 1, batchsize, 26)
            output_logits_backward,
        )
        ### output = {}
        # output["dec_string"] = dec_string
        # output["score_sum"] = score_sum
        # output["position_score"] = position_score
        # output["aaid_sequence"] = aaid_sequence

        bi_score = []
        for output in output_batch:
            bi_score.append(output)
        return bi_score


    def _denovo_decollate_func(self, batch_data: BatchDenovoData) -> List[DenovoData]:
        # Convert the tensor back to numpy if necessary
        spectrum_holder = batch_data.spectrum_holder.numpy()
        
        # Assume that the length of spectrum_holder equals the length of the original data list
        data_list = []
        for i in range(len(spectrum_holder)):
            # Create each DenovoData object from the batch
            denovo_data = DenovoData(
                spectrum_holder=spectrum_holder[i],
                spectrum_original_forward=batch_data.spectrum_original_forward[i],
                spectrum_original_backward=batch_data.spectrum_original_backward[i],
                dia_feature=batch_data.dia_features[i]
            )
            data_list.append(denovo_data)
        
        return data_list

    def _convert2training_data(self, denovo_data : DenovoData, sequence):
        # parse peptide AA sequence to list of ids
        if not sequence:
            logger.info("empty sequence")
        # logger.info(f"dia_feature: {denovo_data.dia_feature.feature_id}, sequence: {sequence}")
        peptide_ids = [deepnovo_config.vocab[x] for x in sequence]
        peptide_ids_forward = peptide_ids[:]
        peptide_ids_forward.insert(0, deepnovo_config.GO_ID)
        peptide_ids_forward.append(deepnovo_config.EOS_ID)
        peptide_ids_backward = peptide_ids[::-1]
        peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
        peptide_ids_backward.append(deepnovo_config.GO_ID)
        prefix_mass = 0.0
        candidate_intensity_forward = []
        spectrum_original_forward = denovo_data.spectrum_original_forward
        spectrum_original_backward = denovo_data.spectrum_original_backward

        for i, id in enumerate(peptide_ids_forward[:-1]):
            prefix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_forward,
                                                          denovo_data.dia_feature.precursor_mass, prefix_mass, 0)
            candidate_intensity_forward.append(candidate_intensity)

        suffix_mass = 0.0
        candidate_intensity_backward=[]
        for i, id in enumerate(peptide_ids_backward[:-1]):
            suffix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_backward,
                                                          denovo_data.dia_feature.precursor_mass, suffix_mass, 1)
            candidate_intensity_backward.append(candidate_intensity)
        return TrainData(denovo_data.spectrum_holder,
                         peptide_ids_forward,
                         peptide_ids_backward,
                         candidate_intensity_forward,
                         candidate_intensity_backward)

    def _sb_beam_search(self, model_wrapper: InferenceModelWrapper, batch_denovo_data: BatchDenovoData,
                        forward_start_point_lists, backward_start_point_lists):
        num_features = len(batch_denovo_data.dia_features)
        top_path_batch_l2r = [[] for _ in range(num_features)]
        top_path_batch_r2l = [[] for _ in range(num_features)]

        direction_cint_map = {Direction.forward: 0, Direction.backward: 1}
       

        # step 1: extract original spectrum
        spectrum_cnn_outputs = model_wrapper.init_spectrum_cnn(batch_denovo_data.spectrum_holder) # (batchszie, 16, 256)

        # initialize activate search list
        active_search_list_l2r = []
        active_search_list_r2l = []

        get_start_mass = lambda x: x.prefix_mass
        first_label = deepnovo_config.GO_ID
        last_label = deepnovo_config.EOS_ID
        for feature_index in range(num_features):  # batchsize
            # all feature in the same batch should be from same direction
            path = SearchPath(
                aa_id_list=[first_label],
                aa_seq_mass=get_start_mass(forward_start_point_lists[feature_index]),
                score_list=[0.0],
                score_sum=0.0,
                direction=Direction.forward,
            )
            search_entry = SearchEntry(
                feature_index=feature_index,
                current_path_list=[path]
            )
            active_search_list_l2r.append(search_entry)
        
        get_start_mass = lambda x: x.suffix_mass
        first_label = deepnovo_config.EOS_ID
        last_label = deepnovo_config.GO_ID
        for feature_index in range(num_features):
            path = SearchPath(
                aa_id_list=[first_label],
                aa_seq_mass=get_start_mass(backward_start_point_lists[feature_index]),
                score_list=[0.0],
                score_sum=0.0,
                direction=Direction.backward,
            )
            search_entry = SearchEntry(
                feature_index=feature_index,
                current_path_list=[path]
            )
            active_search_list_r2l.append(search_entry)

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        current_step = 0
        while True:
            # STEP 2: gather data from active search entries and group into blocks.

            # model input
            block_intensity_input_l2r = []
            # data stored in path
            block_aa_id_list_l2r = []
            block_aa_seq_mass_l2r = []
            block_score_list_l2r = []
            block_score_sum_l2r = []
            block_spectrum_cnn_outputs_l2r = []
            block_knapsack_candidates_l2r = []

            block_intensity_input_r2l = []
            block_aa_id_list_r2l = []
            block_aa_seq_mass_r2l = []
            block_score_list_r2l = []
            block_score_sum_r2l = []
            block_spectrum_cnn_outputs_r2l = []
            block_knapsack_candidates_r2l = []

            # store the number of paths of each search entry in the big blocks
            #     to retrieve the info of each search entry later in STEP 4.
            search_entry_size_l2r = [0] * len(active_search_list_l2r)
            search_entry_size_r2l = [0] * len(active_search_list_r2l)

            last_label = deepnovo_config.EOS_ID
            for entry_index, search_entry in enumerate(active_search_list_l2r):
                # 迭代batchsize
                feature_index = search_entry.feature_index
                current_path_list = search_entry.current_path_list
                precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
                peak_mass_tolerance = forward_start_point_lists[feature_index].mass_tolerance
                spectrum_original = batch_denovo_data.spectrum_original_forward[feature_index]
                # elif direction == Direction.backward:
                #     spectrum_original = batch_denovo_data.spectrum_original_backward[feature_index]
                start_index = len(block_score_sum_l2r)
                for path in current_path_list: # eg: SearchPath(aaid_id_list=[1, 9])
                    aa_id_list = path.aa_id_list
                    aa_id = aa_id_list[-1]
                    score_sum = path.score_sum
                    aa_seq_mass = path.aa_seq_mass
                    score_list = path.score_list

                    # if aa_seq_mass > precursor_mass + peak_mass_tolerance:
                    #     # 终止条件：aa_seq_mass超过了precursor_mass+peak_mass_tolerance
                    #     continue

                    if aa_id == last_label:
                        # 终止条件：最后一个aa是EOS
                        if abs(aa_seq_mass - precursor_mass) <= peak_mass_tolerance:
                            seq = aa_id_list[1:-1]
                            trunc_score_list = score_list[1:-1]
                      
                            top_path_batch_l2r[feature_index].append(
                                BeamSearchedSequence(sequence=seq,
                                                     position_score=trunc_score_list,
                                                     score=path.score_sum / len(seq))
                            )

                    candidate_intensity = get_candidate_intensity(spectrum_original,
                                                                  precursor_mass,
                                                                  aa_seq_mass,
                                                                  direction_cint_map[Direction.forward])
                    residual_mass = precursor_mass - aa_seq_mass - deepnovo_config.mass_ID[last_label]
                    knapsack_tolerance = int(round(peak_mass_tolerance * deepnovo_config.KNAPSACK_AA_RESOLUTION))
                    # list of avaliable aa_id
                    knapsack_candidates = self.knapsack_searcher.search_knapsack(residual_mass, knapsack_tolerance)

                    # if not possible aa, force it to stop.
                    if not knapsack_candidates:
                        if aa_id != last_label and aa_id != deepnovo_config.PAD_ID:
                            knapsack_candidates.append(last_label)
                        else:
                            knapsack_candidates.append(deepnovo_config.PAD_ID)

                    block_intensity_input_l2r.append(candidate_intensity)
                    block_aa_id_list_l2r.append(aa_id_list)
                    block_aa_seq_mass_l2r.append(aa_seq_mass)
                    block_score_list_l2r.append(score_list)
                    block_score_sum_l2r.append(score_sum)
                    block_knapsack_candidates_l2r.append(knapsack_candidates)
                    block_spectrum_cnn_outputs_l2r.append(spectrum_cnn_outputs[entry_index, :, :].unsqueeze_(0))
                    # record the size of each search entry in the blocks
                    search_entry_size_l2r[entry_index] += 1
                # 按score排序
                sorted_indices = sorted(range(start_index, len(block_score_sum_l2r)), key=lambda i: block_score_sum_l2r[i], reverse=True)
                block_intensity_input_l2r[start_index:] = [block_intensity_input_l2r[i] for i in sorted_indices]
                block_aa_id_list_l2r[start_index:] = [block_aa_id_list_l2r[i] for i in sorted_indices]
                block_aa_seq_mass_l2r[start_index:] = [block_aa_seq_mass_l2r[i] for i in sorted_indices]
                block_score_list_l2r[start_index:] = [block_score_list_l2r[i] for i in sorted_indices]
                block_score_sum_l2r[start_index:] = [block_score_sum_l2r[i] for i in sorted_indices]
                block_knapsack_candidates_l2r[start_index:] = [block_knapsack_candidates_l2r[i] for i in sorted_indices]
                block_spectrum_cnn_outputs_l2r[start_index:] = [block_spectrum_cnn_outputs_l2r[i] for i in sorted_indices]      


            last_label = deepnovo_config.GO_ID
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                feature_index = search_entry.feature_index
                current_path_list = search_entry.current_path_list
                precursor_mass = batch_denovo_data.dia_features[feature_index].precursor_mass
                peak_mass_tolerance = backward_start_point_lists[feature_index].mass_tolerance
                spectrum_original = batch_denovo_data.spectrum_original_backward[feature_index]

                start_index = len(block_score_sum_r2l)
                for path in current_path_list:
                    aa_id_list = path.aa_id_list
                    aa_id = aa_id_list[-1]
                    score_sum = path.score_sum
                    aa_seq_mass = path.aa_seq_mass
                    score_list = path.score_list
        
                    # if aa_seq_mass > precursor
                    if aa_id == last_label:
                        if abs(aa_seq_mass - precursor_mass) <= peak_mass_tolerance:
                            seq = aa_id_list[1:-1]
                            trunc_score_list = score_list[1:-1]
                            seq = seq[::-1]
                            trunc_score_list = trunc_score_list[::-1]
                            top_path_batch_r2l[feature_index].append(
                                BeamSearchedSequence(sequence=seq,
                                                     position_score=trunc_score_list,
                                                     score=path.score_sum / len(seq))
                            )
                    candidate_intensity = get_candidate_intensity(spectrum_original,
                                                                  precursor_mass,
                                                                  aa_seq_mass,
                                                                  direction_cint_map[Direction.backward])
                    residual_mass = precursor_mass - aa_seq_mass - deepnovo_config.mass_ID[last_label]
                    knapsack_tolerance = int(round(peak_mass_tolerance * deepnovo_config.KNAPSACK_AA_RESOLUTION))
                    # list of avaliable aa_id
                    knapsack_candidates = self.knapsack_searcher.search_knapsack(residual_mass, knapsack_tolerance)
                    # if not possible aa, force it to stop.
                    if not knapsack_candidates:
                        if aa_id != last_label and aa_id != deepnovo_config.PAD_ID:
                            knapsack_candidates.append(last_label)
                        else:
                            knapsack_candidates.append(deepnovo_config.PAD_ID)

                    block_intensity_input_r2l.append(candidate_intensity)
                    block_aa_id_list_r2l.append(aa_id_list)
                    block_aa_seq_mass_r2l.append(aa_seq_mass)
                    block_score_list_r2l.append(score_list)
                    block_score_sum_r2l.append(score_sum)
                    block_knapsack_candidates_r2l.append(knapsack_candidates)
                    block_spectrum_cnn_outputs_r2l.append(spectrum_cnn_outputs[entry_index, :, :].unsqueeze_(0))
                    # record the size of each search entry in the blocks
                    search_entry_size_r2l[entry_index] += 1
                # 按score排序
                sorted_indices = sorted(range(start_index, len(block_score_sum_r2l)), key=lambda i: block_score_sum_r2l[i], reverse=True)
                block_intensity_input_r2l[start_index:] = [block_intensity_input_r2l[i] for i in sorted_indices]
                block_aa_id_list_r2l[start_index:] = [block_aa_id_list_r2l[i] for i in sorted_indices]
                block_aa_seq_mass_r2l[start_index:] = [block_aa_seq_mass_r2l[i] for i in sorted_indices]
                block_score_list_r2l[start_index:] = [block_score_list_r2l[i] for i in sorted_indices]
                block_score_sum_r2l[start_index:] = [block_score_sum_r2l[i] for i in sorted_indices]
                block_knapsack_candidates_r2l[start_index:] = [block_knapsack_candidates_r2l[i] for i in sorted_indices]
                block_spectrum_cnn_outputs_r2l[start_index:] = [block_spectrum_cnn_outputs_r2l[i] for i in sorted_indices]


            # 检查 block_aa_id_list_l2r 和 block_aa_id_list_r2l 中所有序列是否都包含终止符
            if all(deepnovo_config.EOS_ID in x for x in block_aa_id_list_l2r) and all(
                deepnovo_config.GO_ID in x for x in block_aa_id_list_r2l):
                break

            # step 3 run model on data blocks to predict next AA.
           
            # 逐元素比较两个列表
            assert all(l2r == r2l for l2r, r2l in zip(search_entry_size_l2r, search_entry_size_r2l))
            # (batchsize * beamsize, 26, 40, 10)
            block_intensity_input_l2r = torch.from_numpy(np.array(block_intensity_input_l2r)).to(device)
            # (seq_len, batchsize * beamsize)
            block_decoder_inputs_l2r = torch.from_numpy(np.array(block_aa_id_list_l2r).transpose(1, 0)).to(device)
            block_intensity_input_r2l = torch.from_numpy(np.array(block_intensity_input_r2l)).to(device)
            block_decoder_inputs_r2l = torch.from_numpy(np.array(block_aa_id_list_r2l).transpose(1, 0)).to(device)

            block_spectrum_cnn_outputs_l2r = torch.cat(block_spectrum_cnn_outputs_l2r, dim=0)
                
            current_log_prob_forward, current_log_prob_backward = model_wrapper.inference(
                block_spectrum_cnn_outputs_l2r,
                block_intensity_input_l2r,
                block_intensity_input_r2l,
                block_decoder_inputs_l2r,
                block_decoder_inputs_r2l
            )
            
            Logsoftmax = torch.nn.LogSoftmax(dim=1)
            current_log_prob_forward = Logsoftmax(current_log_prob_forward)
            current_log_prob_backward = Logsoftmax(current_log_prob_backward)
            # transfer log_prob back to cpu
            current_log_prob_forward = current_log_prob_forward.cpu().numpy()
            current_log_prob_backward = current_log_prob_backward.cpu().numpy()


            # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
                        # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list_l2r):
                new_path_list = []
                direction = search_entry.current_path_list[0].direction
                for index in range(block_index, block_index + search_entry_size_l2r[entry_index]):
                    for aa_id in block_knapsack_candidates_l2r[index]:  # < batchsize * beamsize
                        if aa_id > 2:
                            # do not add score of GO, EOS, PAD
                            new_score_list = block_score_list_l2r[index] + [current_log_prob_forward[index][aa_id]]
                            new_score_sum = block_score_sum_l2r[index] + current_log_prob_forward[index][aa_id]
                        else:
                            new_score_list = block_score_list_l2r[index] + [0.0]
                            new_score_sum = block_score_sum_l2r[index] + 0.0

                        new_path = SearchPath(
                            aa_id_list=block_aa_id_list_l2r[index] + [aa_id],
                            aa_seq_mass=block_aa_seq_mass_l2r[index] + deepnovo_config.mass_ID[aa_id],
                            score_list=new_score_list,
                            score_sum=new_score_sum,
                            direction=direction
                        )
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x.score_sum for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size]
                    search_entry.current_path_list = [new_path_list[ii] for ii in top_k_index]  # list of SearchPath
                else:
                    search_entry.current_path_list = new_path_list

                block_index += search_entry_size_l2r[entry_index]
            # remove search entry with no path
            active_search_list_l2r = [x for x in active_search_list_l2r if x.current_path_list]  # batchsize

            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list_r2l):
                new_path_list = []
                direction = search_entry.current_path_list[0].direction
                for index in range(block_index, block_index + search_entry_size_r2l[entry_index]):
                    for aa_id in block_knapsack_candidates_r2l[index]:
                        if aa_id > 2:
                            new_score_list = block_score_list_r2l[index] + [current_log_prob_backward[index][aa_id]]
                            new_score_sum = block_score_sum_r2l[index] + current_log_prob_backward[index][aa_id]
                        else:
                            new_score_list = block_score_list_r2l[index] + [0.0]
                            new_score_sum = block_score_sum_r2l[index] + 0.0

                        new_path = SearchPath(
                            aa_id_list=block_aa_id_list_r2l[index] + [aa_id],
                            aa_seq_mass=block_aa_seq_mass_r2l[index] + deepnovo_config.mass_ID[aa_id],
                            score_list=new_score_list,
                            score_sum=new_score_sum,
                            direction=direction
                        )
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x.score_sum for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size]
                    search_entry.current_path_list = [new_path_list[ii] for ii in top_k_index]  # list of SearchPath
                else:
                    search_entry.current_path_list = new_path_list
                block_index += search_entry_size_r2l[entry_index]
            # remove search entry with no path
            active_search_list_r2l = [x for x in active_search_list_r2l if x.current_path_list]

            current_step += 1
            if not active_search_list_l2r and not active_search_list_r2l:
                break

        return top_path_batch_l2r, top_path_batch_r2l