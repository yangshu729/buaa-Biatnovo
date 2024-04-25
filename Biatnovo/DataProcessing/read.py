import math
import os
import random
import gc
import sys
import traceback
import deepnovo_config
from Biatnovo.DataProcessing.deepnovo_worker_io import WorkerI
from functools import partial
from multiprocessing import Pool
from six.moves import xrange  # pylint: disable=redefined-builtin
from DataProcess.deepnovo_cython_modules import get_candidate_intensity


# 随机选择验证的500行feature进行验证
def read_random_stack(worker_io, feature_index_list, stack_size, opt, training_mode=False, step=0):
    """TODO(nh2tran): docstring."""

    print("read_random_stack()")
    if not training_mode:
        # if validation
        random_index_list = random.sample(feature_index_list, min(stack_size, len(feature_index_list)))
    else:
        # not random, but in order
        if (step + 1) * stack_size > len(feature_index_list):
            random_index_list = feature_index_list[step * stack_size :]
        else:
            random_index_list = feature_index_list[step * stack_size : (step + 1) * stack_size]
    # print("random index list:", random_index_list)
    return read_spectra(worker_io, random_index_list, opt)


# feature_index 第几个，worker_i 对象，feature_fr 特征文件 spectrum_fr 谱图文件
def read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr, opt):
    # 应该是返回feature index 形成的5个？谱图信息
    # spectrum = {"feature_id": feature_id,  # str(feature_index),#scan,
    #             "feature_area": feature_area,
    #             "raw_sequence": raw_sequence,
    #             "precursor_mass": precursor_mass,
    #             "spectrum_holder": spectrum_holder,
    #             "spectrum_original_forward": spectrum_original_forward,
    #             "spectrum_original_backward": spectrum_original_backward,
    #             "precursor_mz": precursor_mz,
    #             "precursor_charge": precursor_charge,
    #             "scan_list_middle": scan_list_middle,
    #             "scan_list_original": scan_list_original,
    #             "ms1_profile": ms1_profile}
    spectrum_list = worker_i.get_spectrum([feature_index], feature_fr, spectrum_fr)
    if not spectrum_list:
        return None
    spectrum = spectrum_list[0]
    feature_id = spectrum["feature_id"]
    raw_sequence = spectrum["raw_sequence"]
    precursor_mass = spectrum["precursor_mass"]

    spectrum_holder = spectrum["spectrum_holder"] if opt.use_lstm else None
    spectrum_original_forward = spectrum["spectrum_original_forward"]
    spectrum_original_backward = spectrum["spectrum_original_backward"]
    # spectrum_holder size (5, 150000) (self.neighbor_size, self.MZ_SIZE)
    # spectrum_original_forward size (5, 150000)
    # spectrum_original_backward size (5, 150000)

    ### parse peptide sequence
    # unlabelled spectra with empty raw_sequence can be used as neighbors,
    #   but not as main spectrum for training >> skip empty raw_sequence
    if not raw_sequence:
        status = "empty"
        return None, None, status
    # parse peptide sequence, skip if unknown_modification
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    unknown_modification = False
    # 对肽段进行修饰 质量加
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
                # ~ elif ("".join(raw_sequence[index:index+8])=="(+42.01)"):
                # ~ print("ERROR: unknown modification!")
                # ~ print("raw_sequence = ", raw_sequence)
                # ~ sys.exit()
                unknown_modification = True
                break
        else:
            peptide.append(raw_sequence[index])
            index += 1
    if unknown_modification:
        status = "mod"
        return None, None, status
    # skip if peptide length > MAX_LEN (train: 30; decode:50)
    peptide_len = len(peptide)
    if peptide_len > deepnovo_config.MAX_LEN:
        status = "length"
        return None, None, status

    # all mass and sequence filters passed
    # counter_read += 1
    # 选择最近的bucket_id
    ### prepare forward, backward, and padding
    for bucket_id, target_size in enumerate(deepnovo_config._buckets):
        if peptide_len + 2 <= target_size:  # +2 to include GO and EOS
            break
    decoder_size = deepnovo_config._buckets[bucket_id]
    # parse peptide AA sequence to list of ids
    # peptide_ids为对肽段每个字母的编码，用id进行代替 按照config中定义的顺序
    peptide_ids = [deepnovo_config.vocab[x] for x in peptide]
    # padding
    # 长度短的肽段进行补齐，补0
    pad_size = decoder_size - (len(peptide_ids) + 2)
    # forward, 进行padding
    if opt.direction == 0 or opt.direction == 2:
        peptide_ids_forward = peptide_ids[:]
        peptide_ids_forward.insert(0, deepnovo_config.GO_ID)
        peptide_ids_forward.append(deepnovo_config.EOS_ID)
        peptide_ids_forward += [deepnovo_config.PAD_ID] * pad_size
    # backward
    if opt.direction == 1 or opt.direction == 2:
        peptide_ids_backward = peptide_ids[::-1]
        peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
        peptide_ids_backward.append(deepnovo_config.GO_ID)
        peptide_ids_backward += [deepnovo_config.PAD_ID] * pad_size

    ### retrieve candidate_intensity for test/decode_true_feeding
    # 默认不打开beam search选项
    if not opt.beam_search:
        # forward
        if opt.direction == 0 or opt.direction == 2:
            candidate_intensity_list_forward = []
            prefix_mass = 0.0
            # 针对每一个肽段的字母
            for index in xrange(decoder_size):
                # 计算前缀质量
                prefix_mass += deepnovo_config.mass_ID[peptide_ids_forward[index]]
                # print("prefix_mass:", prefix_mass)
                # 变为（26， 40， 10），返回precursor质量以及prefix质量对应的候选强度值
                candidate_intensity = get_candidate_intensity(
                    spectrum_original_forward, precursor_mass, prefix_mass, 0  # (5, 150000)
                )
                candidate_intensity_list_forward.append(candidate_intensity)
        # backward
        if opt.direction == 1 or opt.direction == 2:
            candidate_intensity_list_backward = []
            suffix_mass = 0.0
            for index in xrange(decoder_size):
                suffix_mass += deepnovo_config.mass_ID[peptide_ids_backward[index]]
                candidate_intensity = get_candidate_intensity(
                    spectrum_original_backward, precursor_mass, suffix_mass, 1
                )
                candidate_intensity_list_backward.append(candidate_intensity)
    # length(candidate_intensity_list_backward) = 12
    # length(candidate_intensity_list_backward[0]) = 26
    # length(candidate_intensity_list_backward[0][0]) = 40
    # length(candidate_intensity_list_backward[0][0][0]) = 10
    ### assign data to buckets
    if opt.beam_search:
        if opt.direction == 0:
            data = [feature_id, spectrum_holder, spectrum_original_forward, precursor_mass, peptide_ids_forward]

        elif opt.direction == 1:
            data = [feature_id, spectrum_holder, spectrum_original_backward, precursor_mass, peptide_ids_backward]

        else:
            data = [
                feature_id,
                spectrum_holder,
                spectrum_original_forward,
                spectrum_original_backward,
                precursor_mass,
                peptide_ids_forward,
                peptide_ids_backward,
            ]

    else:
        if opt.direction == 0:
            data = [spectrum_holder, candidate_intensity_list_forward, peptide_ids_forward]
        elif opt.direction == 1:
            data = [spectrum_holder, candidate_intensity_list_backward, peptide_ids_backward]
        else:
            data = [
                spectrum_holder,
                candidate_intensity_list_forward,
                candidate_intensity_list_backward,
                peptide_ids_forward,
                peptide_ids_backward,
            ]
    # 这个bucket_id还不知道是什么 肽段的长度为12 22 等等
    return data, bucket_id, "OK"


# feature_index 第几个，worker_i 对象，feature_fr 特征文件 spectrum_fr 谱图文件
def read_single_spectrum_true_feeding(spectrum_batch, peptide_sequence, opt):
    # 应该是返回feature index 形成的5个？谱图信息
    # spectrum = {"feature_id": feature_id,  # str(feature_index),#scan,
    #             "feature_area": feature_area,
    #             "raw_sequence": raw_sequence,
    #             "precursor_mass": precursor_mass,
    #             "spectrum_holder": spectrum_holder,
    #             "spectrum_original_forward": spectrum_original_forward,
    #             "spectrum_original_backward": spectrum_original_backward,
    #             "precursor_mz": precursor_mz,
    #             "precursor_charge": precursor_charge,
    #             "scan_list_middle": scan_list_middle,
    #             "scan_list_original": scan_list_original,
    #             "ms1_profile": ms1_profile}
    if not spectrum_batch:
        return None
    spectrum = spectrum_batch
    feature_id = spectrum["feature_id"]
    precursor_mass = spectrum["precursor_mass"]

    spectrum_holder = spectrum["spectrum_holder"]
    spectrum_original_forward = spectrum["spectrum_original_forward"]
    spectrum_original_backward = spectrum["spectrum_original_backward"]

    peptide = peptide_sequence

    # skip if peptide length > MAX_LEN (train: 30; decode:50)
    peptide_len = len(peptide)
    if peptide_len > deepnovo_config.MAX_LEN:
        status = "length"
        return None, None, status

    # all mass and sequence filters passed
    # counter_read += 1
    # 选择最近的bucket_id
    ### prepare forward, backward, and padding
    for bucket_id, target_size in enumerate(deepnovo_config._buckets):
        if peptide_len + 2 <= target_size:  # +2 to include GO and EOS
            break
    decoder_size = deepnovo_config._buckets[bucket_id]
    # parse peptide AA sequence to list of ids
    # peptide_ids为对肽段每个字母的编码，用id进行代替 按照config中定义的顺序
    peptide_ids = [deepnovo_config.vocab[x] for x in peptide]
    # padding
    # 长度短的肽段进行补齐，补0
    pad_size = decoder_size - (len(peptide_ids) + 2)
    # forward, 进行padding
    if opt.direction == 0 or opt.direction == 2:
        peptide_ids_forward = peptide_ids[:]
        peptide_ids_forward.insert(0, deepnovo_config.GO_ID)
        peptide_ids_forward.append(deepnovo_config.EOS_ID)
        peptide_ids_forward += [deepnovo_config.PAD_ID] * pad_size
    # backward
    if opt.direction == 1 or opt.direction == 2:
        peptide_ids_backward = peptide_ids[::-1]
        peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
        peptide_ids_backward.append(deepnovo_config.GO_ID)
        peptide_ids_backward += [deepnovo_config.PAD_ID] * pad_size

    ### retrieve candidate_intensity for test/decode_true_feeding
    # 默认不打开beam search选项
    if not opt.beam_search:
        # forward
        if opt.direction == 0 or opt.direction == 2:
            candidate_intensity_list_forward = []
            prefix_mass = 0.0
            # 针对每一个肽段的字母
            for index in xrange(decoder_size):
                # 计算前缀质量
                prefix_mass += deepnovo_config.mass_ID[peptide_ids_forward[index]]
                # print("prefix_mass:", prefix_mass)
                # 变为（26， 40， 10），返回precursor质量以及prefix质量对应的候选强度值
                candidate_intensity = get_candidate_intensity(
                    spectrum_original_forward, precursor_mass, prefix_mass, 0  # (5, 150000)
                )
                candidate_intensity_list_forward.append(candidate_intensity)
        # backward
        if opt.direction == 1 or opt.direction == 2:
            candidate_intensity_list_backward = []
            suffix_mass = 0.0
            for index in xrange(decoder_size):
                suffix_mass += deepnovo_config.mass_ID[peptide_ids_backward[index]]
                candidate_intensity = get_candidate_intensity(
                    spectrum_original_backward, precursor_mass, suffix_mass, 1
                )
                candidate_intensity_list_backward.append(candidate_intensity)
    # length(candidate_intensity_list_backward) = 12
    # length(candidate_intensity_list_backward[0]) = 26
    # length(candidate_intensity_list_backward[0][0]) = 40
    # length(candidate_intensity_list_backward[0][0][0]) = 10
    ### assign data to buckets
    if opt.beam_search:
        if opt.direction == 0:
            data = [feature_id, spectrum_holder, spectrum_original_forward, precursor_mass, peptide_ids_forward]

        elif opt.direction == 1:
            data = [feature_id, spectrum_holder, spectrum_original_backward, precursor_mass, peptide_ids_backward]

        else:
            data = [
                feature_id,
                spectrum_holder,
                spectrum_original_forward,
                spectrum_original_backward,
                precursor_mass,
                peptide_ids_forward,
                peptide_ids_backward,
            ]

    else:
        if opt.direction == 0:
            data = [spectrum_holder, candidate_intensity_list_forward, peptide_ids_forward]
        elif opt.direction == 1:
            data = [spectrum_holder, candidate_intensity_list_backward, peptide_ids_backward]
        else:
            data = [
                spectrum_holder,
                candidate_intensity_list_forward,
                candidate_intensity_list_backward,
                peptide_ids_forward,
                peptide_ids_backward,
            ]
    # 这个bucket_id还不知道是什么 肽段的长度为12 22 等等
    return data, bucket_id, "OK"


def _prepare_data(feature_index, worker_i, opt):
    """

    :param feature_index:
    :param get_spectrum: a callable, takes in [feature)index] and result spectrum_list
    :return: None if the input feature is not valid
    data, bucket_id, status_code
    """
    ### retrieve spectrum information
    # read spectrum, skip if precursor_mass > MZ_MAX, pre-process spectrum
    try:
        with open(worker_i.input_feature_file, "r") as feature_fr:
            with open(worker_i.input_spectrum_file, "r") as spectrum_fr:
                return read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr, opt)
    except Exception:
        print("exception in _prepare_data: ")
        traceback.print_exc()
        raise


# 返回每一种肽段长度的数据
# data = [spectrum_holder,
#         #               candidate_intensity_list_forward,
#         #               candidate_intensity_list_backward,
#         #               peptide_ids_forward,
#         #               peptide_ids_backward,
#         #               ms1_profile]
def read_spectra(worker_io, feature_index_list, opt):
    """TODO(nh2tran): docstring."""

    print("".join(["="] * 80))  # section-separating line
    print("read_spectra()")
    # 根据肽段长度将谱图划分为bucket
    # assign spectrum into buckets according to peptide length
    data_set = [[] for _ in deepnovo_config._buckets]

    # use single/multi processor to read data during training
    worker_i = WorkerI(worker_io)
    if opt.multiprocessor == 1:
        with open(worker_i.input_feature_file, "r") as feature_fr:
            with open(worker_i.input_spectrum_file, "r") as spectrum_fr:
                # 读第feature_index个单个的图谱，遍历feature文件，每一行feature形成
                #       data = [spectrum_holder,
                #               candidate_intensity_list_forward,
                #               candidate_intensity_list_backward,
                #               peptide_ids_forward,
                #               peptide_ids_backward,
                #               ms1_profile]
                result_list = [
                    read_single_spectrum(feature_index, worker_i, feature_fr, spectrum_fr, opt)
                    for feature_index in feature_index_list
                ]
    else:
        mp_func = partial(_prepare_data, worker_i=worker_i, opt=opt)
        gc.collect()
        pool = Pool(processes=opt.multiprocessor)
        try:
            result_list = pool.map_async(mp_func, feature_index_list).get(9999)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(1)
    # 统计肽段序列为空的特征有多少
    counter = len(feature_index_list)
    # worker_io is designed for both prediction and training, hence it does not
    #   check raw_sequence for empty/mod/len because raw_sequence is only provided
    #   in training.
    counter_skipped_empty = 0
    counter_skipped_mod = 0
    counter_skipped_len = 0
    counter_read = 0
    # ~ counter_skipped_mass_precision = 0
    for result in result_list:
        if result is None:
            continue
        # 是由read_single_spectra函数返回的
        data, bucket_id, status = result
        if data:
            counter_read += 1
            data_set[bucket_id].append(data)
        elif status == "empty":
            counter_skipped_empty += 1
        elif status == "mod":
            counter_skipped_mod += 1
        elif status == "length":
            counter_skipped_len += 1
    worker_io.feature_count["read"] += len(result_list)

    del result_list
    del worker_i
    gc.collect()

    counter_skipped_mass = worker_io.feature_count["skipped_mass"]
    counter_skipped = counter_skipped_mass + counter_skipped_empty + counter_skipped_mod + counter_skipped_len
    print("  total peptide %d" % counter)
    print("    peptide read %d" % counter_read)
    print("    peptide skipped %d" % counter_skipped)
    print("    peptide skipped by mass %d" % counter_skipped_mass)
    print("    peptide skipped by empty %d" % counter_skipped_empty)
    print("    peptide skipped by mod %d" % counter_skipped_mod)
    print("    peptide skipped by len %d" % counter_skipped_len)
    # ~ print(counter_skipped_mass_precision)
    # ~ print(abc)

    return data_set, counter_read
