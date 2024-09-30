import csv
import logging
import os
import pickle
import re
from typing import List
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from DataProcess.deepnovo_cython_modules import process_spectrum,get_candidate_intensity
import deepnovo_config
logger = logging.getLogger(__name__)

@dataclass
class DIAFeature:
    feature_id : str
    feature_area : float
    precursor_mz : float
    precursor_charge : int
    precursor_mass : float
    rt_mean :  float
    peptide : list
    scan_list_middle: list = None
    scan_list : list = None
    ms1_list : list = None

@dataclass
class DenovoData:
    spectrum_holder : np.ndarray
    spectrum_original_forward : np.ndarray
    spectrum_original_backward : np.ndarray
    dia_feature : DIAFeature

@dataclass
class BatchDenovoData:
    spectrum_holder : torch.Tensor
    spectrum_original_forward : List[np.ndarray]
    spectrum_original_backward : List[np.ndarray]
    dia_features : List[DIAFeature]

@dataclass
class TrainData:
    spectrum_holder: np.ndarray
    peptide_ids_forward: list
    peptide_ids_backward: list
    forward_candidate_intensity: list
    backward_candidate_intensity: list

def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide

class DeepNovoTrainDataset(Dataset):
    def __init__(self, feature_filename, spectrum_filename, transform=None):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        logger.info(f"input spectrum file: {spectrum_filename}")
        logger.info(f"input feature file: {feature_filename}")
        self.spectrum_filename = spectrum_filename
        self.feature_filename = feature_filename
        self.input_spectrum_handle = None
        self.feature_list = []
        self.spectrum_location_dict = {}
        self.spectrum_rtinseconds_dict = {}
        self.spectrum_count = 0
        self.input_feature_handle = open(self.feature_filename, "r")
        self.transform = transform
        # record the status of spectra that have been read
        self.feature_count = {"total": 0, "read": 0, "skipped": 0, "skipped_mass": 0}
        ### store file location of each spectrum for random access {scan:location}
        ### since mgf file can be rather big, cache the locations for each spectrum mgf file.
        spectrum_location_file = self.spectrum_filename + ".locations.pkl"
        # 读取缓存里面的内容
        if os.path.exists(spectrum_location_file):
            print("WorkerIO: read cached spectrum locations")
            with open(spectrum_location_file, "rb") as fr:
                data = pickle.load(fr)
                (
                    self.spectrum_location_dict,
                    self.spectrum_rtinseconds_dict,
                    self.spectrum_count,
                ) = data
        else:
            # 读取mgf文件是自己写的按照行读取
            print("WorkerIO: build spectrum location from scratch")
            spectrum_location_dict = {}
            spectrum_rtinseconds_dict = {}
            line = True
            while line:
                current_location = self.input_spectrum_handle.tell()
                line = self.input_spectrum_handle.readline()
                if "BEGIN IONS" in line:
                    spectrum_location = current_location
                elif "SCANS=" in line:
                    # scan = re.split('=|\r\n', line)[1]
                    scan = re.split("=|\n", line)[1]
                    spectrum_location_dict[scan] = spectrum_location
                elif "RTINSECONDS=" in line:
                    # rtinseconds = float(re.split('=|\r\n', line)[1])
                    rtinseconds = float(re.split("=|\n", line)[1])
                    spectrum_rtinseconds_dict[scan] = rtinseconds
            self.spectrum_location_dict = spectrum_location_dict
            self.spectrum_rtinseconds_dict = spectrum_rtinseconds_dict
            self.spectrum_count = len(spectrum_location_dict)
            with open(spectrum_location_file, "wb") as fw:
                pickle.dump(
                    (
                        self.spectrum_location_dict,
                        self.spectrum_rtinseconds_dict,
                        self.spectrum_count,
                    ),
                    fw,
                )
        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        with open(feature_filename, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(deepnovo_config.col_feature_id)
            mz_index = header.index(deepnovo_config.col_precursor_mz)
            z_index = header.index(deepnovo_config.col_precursor_charge)
            rt_mean_index = header.index(deepnovo_config.col_rt_mean)
            seq_index = header.index(deepnovo_config.col_raw_sequence)
            scan_index = header.index(deepnovo_config.col_scan_list)
            feature_area_index = header.index(deepnovo_config.col_feature_area)
            ms1_index = header.index(deepnovo_config.col_ms1_list)
            for line in reader:
                mass = (float(line[mz_index]) - deepnovo_config.mass_H) * float(line[z_index])
                ok, peptide = parse_raw_sequence(line[seq_index])
                if not ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > deepnovo_config.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index]} skipped by mass, mass is {mass}")
                    continue
                if len(peptide) >= deepnovo_config.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                feature_id = line[feature_id_index]
                feature_area_str = line[feature_area_index]
                feature_area = float(feature_area_str) if feature_area_str else 1.0
                precursor_mz = float(line[mz_index])
                precursor_charge = float(line[z_index])
                rt_mean = float(line[rt_mean_index])
                scan_list = re.split(";", line[scan_index])
                ms1_list = re.split(";", line[ms1_index])
                if (len(scan_list) != len(ms1_list)):
                    ms1_list = ["1:1"] * len(scan_list)  # mock ms1_list数据
                assert len(scan_list) == len(ms1_list), "Error: scan_list and ms1_list not matched."
                new_feature = DIAFeature(feature_id, feature_area, precursor_mz,
                                         precursor_charge, mass, rt_mean, peptide, scan_list = scan_list, ms1_list = ms1_list)
                self.feature_list.append(new_feature)
        logger.info(f"skipped_by_mass: {skipped_by_mass}, skipped_by_ptm: {skipped_by_ptm}, skipped_by_length: {skipped_by_length}")
        logger.info(f"total features: {len(self.feature_list)}")

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_spectrum(feature, self.input_spectrum_handle)

    def _get_spectrum(self, feature : DIAFeature, input_spectrum_file_handle):
        """TODO(nh2tran): docstring."""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerIO: get_spectrum()")

        # parse and process spectrum
        (
            spectrum_holder,
            spectrum_original_forward,
            spectrum_original_backward,
            scan_list_middle,
            scan_list_original,
            ms1_profile,
        ) = self._parse_spectrum(
            feature.precursor_mz,
            feature.precursor_mass,
            feature.rt_mean,
            feature.scan_list,
            feature.ms1_list,
            input_spectrum_file_handle,
        )

        # parse peptide AA sequence to list of ids
        peptide_ids = [deepnovo_config.vocab[x] for x in feature.peptide]
        peptide_ids_forward = peptide_ids[:]
        peptide_ids_forward.insert(0, deepnovo_config.GO_ID)
        peptide_ids_forward.append(deepnovo_config.EOS_ID)
        peptide_ids_backward = peptide_ids[::-1]
        peptide_ids_backward.insert(0, deepnovo_config.EOS_ID)
        peptide_ids_backward.append(deepnovo_config.GO_ID)
        prefix_mass = 0.0
        candidate_intensity_forward = []
        for i, id in enumerate(peptide_ids_forward[:-1]):
            prefix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_forward, feature.precursor_mass, prefix_mass, 0)
            candidate_intensity_forward.append(candidate_intensity)

        suffix_mass = 0.0
        candidate_intensity_backward=[]
        for i, id in enumerate(peptide_ids_backward[:-1]):
            suffix_mass += deepnovo_config.mass_ID[id]
            candidate_intensity = get_candidate_intensity(spectrum_original_backward, feature.precursor_mass, suffix_mass, 1)
            candidate_intensity_backward.append(candidate_intensity)
        return TrainData(spectrum_holder,
                         peptide_ids_forward,
                         peptide_ids_backward,
                         candidate_intensity_forward,
                         candidate_intensity_backward)


    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def _parse_spectrum(self, precursor_mz, precursor_mass, rt_mean, scan_list, ms1_list, input_file_handle):
        """TODO(nh2tran): docstring."""

        #~ print("".join(["="] * 80)) # section-separating line
        #~ print("WorkerIO: _parse_spectrum()")

        spectrum_holder_list = []
        spectrum_original_forward_list = []
        spectrum_original_backward_list = []

        ### select best neighbors from the scan_list by their distance to rt_mean
        # probably move this selection to get_location(), run once rather than repeating
        neighbor_count = len(scan_list)
        best_scan_index = None
        best_distance = float('inf')
        for scan_index, scan in enumerate(scan_list):
            distance = abs(self.spectrum_rtinseconds_dict[scan] - rt_mean)
            if distance < best_distance:
                best_distance = distance
                best_scan_index = scan_index
        neighbor_center = best_scan_index
        neighbor_left_count = neighbor_center
        neighbor_right_count = neighbor_count - neighbor_left_count - 1
        neighbor_size_half = deepnovo_config.neighbor_size // 2
        neighbor_left_count = min(neighbor_left_count, neighbor_size_half)
        neighbor_right_count = min(neighbor_right_count, neighbor_size_half)

        ### padding zero arrays to the left if not enough neighbor spectra
        if neighbor_left_count < neighbor_size_half:
            for x in range(neighbor_size_half - neighbor_left_count):
                spectrum_holder_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))
                spectrum_original_forward_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))
                spectrum_original_backward_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))

        ### parse and add neighbor spectra
        scan_list_middle = []
        ms1_intensity_list_middle = []
        for index in range(neighbor_center - neighbor_left_count, neighbor_center + neighbor_right_count + 1):
            scan = scan_list[index]
            scan_list_middle.append(scan)
            ms1_entry = ms1_list[index]
            ms1_intensity = float(re.split(':', ms1_entry)[1])
            ms1_intensity_list_middle.append(ms1_intensity)
            ms1_intensity_max = max(ms1_intensity_list_middle)
        ms1_intensity_max = 1.0
        assert ms1_intensity_max > 0.0, "Error: Zero ms1_intensity_max"
        ms1_intensity_list_middle = [x/ms1_intensity_max for x in ms1_intensity_list_middle]
        for scan, ms1_intensity in zip(scan_list_middle, ms1_intensity_list_middle):
            spectrum_location = self.spectrum_location_dict[scan]
            input_file_handle.seek(spectrum_location)
            # parse header lines
            line = input_file_handle.readline()
            assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
            line = input_file_handle.readline()
            assert "TITLE=" in line, "Error: wrong input TITLE="
            line = input_file_handle.readline()
            assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
            line = input_file_handle.readline()
            assert "CHARGE=" in line, "Error: wrong input CHARGE="
            line = input_file_handle.readline()
            assert "SCANS=" in line, "Error: wrong input SCANS="
            line = input_file_handle.readline()
            assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
            # parse fragment ions
            mz_list, intensity_list = self._parse_spectrum_ion(input_file_handle)
            # pre-process spectrum
            (spectrum_holder,
            spectrum_original_forward,
            spectrum_original_backward) = process_spectrum(mz_list,
                                                            intensity_list,
                                                            precursor_mass)
            # normalize by each individual spectrum
            #~ spectrum_holder /= np.max(spectrum_holder)
            #~ spectrum_original_forward /= np.max(spectrum_original_forward)
            #~ spectrum_original_backward /= np.max(spectrum_original_backward)
            # weight by ms1 profile
            #~ spectrum_holder *= ms1_intensity
            #~ spectrum_original_forward *= ms1_intensity
            #~ spectrum_original_backward *= ms1_intensity
            # add spectrum to the neighbor list
            spectrum_holder_list.append(spectrum_holder)
            spectrum_original_forward_list.append(spectrum_original_forward)
            spectrum_original_backward_list.append(spectrum_original_backward)
        ### padding zero arrays to the right if not enough neighbor spectra
        if neighbor_right_count < neighbor_size_half:
            for x in range(neighbor_size_half - neighbor_right_count):
                spectrum_holder_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))
                spectrum_original_forward_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))
                spectrum_original_backward_list.append(np.zeros(
                    shape=(1, deepnovo_config.MZ_SIZE),
                    dtype=np.float32))

        spectrum_holder = np.vstack(spectrum_holder_list)
        spectrum_original_forward = np.vstack(spectrum_original_forward_list)
        spectrum_original_backward = np.vstack(spectrum_original_backward_list)
        assert spectrum_holder.shape == (deepnovo_config.neighbor_size,
                                         deepnovo_config.MZ_SIZE), "Error:shape"
        # spectrum-CNN normalization: by feature
        spectrum_holder /= np.max(spectrum_holder)

        # ms1_profile
        for x in range(neighbor_size_half - neighbor_left_count):
            ms1_intensity_list_middle = [0.0] + ms1_intensity_list_middle
        for x in range(neighbor_size_half - neighbor_right_count):
            ms1_intensity_list_middle = ms1_intensity_list_middle + [0.0]
        assert len(ms1_intensity_list_middle) == deepnovo_config.neighbor_size, "Error: ms1 profile"
        ms1_profile = np.array(ms1_intensity_list_middle)

        return spectrum_holder, spectrum_original_forward, spectrum_original_backward, scan_list_middle, scan_list, ms1_profile


    def _parse_spectrum_ion(self, input_file_handle):
        """TODO(nh2tran): docstring."""

        #~ print("".join(["="] * 80)) # section-separating line
        #~ print("WorkerIO: _parse_spectrum_ion()")

        # ion
        mz_list = []
        intensity_list = []
        line = input_file_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > deepnovo_config.MZ_MAX:
                line = input_file_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = input_file_handle.readline()

        return mz_list, intensity_list
    

def collate_func(train_data_list: list[TrainData]):
    """

    :param train_data_list: list of TrainData
    :return:
    """
    
    #train_data_list.sort(key=lambda x: len(x.peptide_ids_forward), reverse=True)
    # batch_max_seq_len = len(train_data_list[0].peptide_ids_forward)
    batch_max_seq_len = max([len(x.peptide_ids_forward) for x in train_data_list])
    intensity_shape = train_data_list[0].forward_candidate_intensity[0].shape
    spectrum_holder = [x.spectrum_holder for x in train_data_list]
    spectrum_holder = np.stack(spectrum_holder) # [batch_size, neibor, mz_size]
    spectrum_holder = torch.from_numpy(spectrum_holder)

    batch_forward_intensity = []
    batch_peptide_ids_forward = []

    for data in train_data_list:
        f_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        forward_intensity = np.stack(data.forward_candidate_intensity)
        f_intensity[:forward_intensity.shape[0], :, :, :] = forward_intensity
        batch_forward_intensity.append(f_intensity)
        f_peptide = np.zeros((batch_max_seq_len,), np.int64)
        forward_peptide_ids = np.array(data.peptide_ids_forward, np.int64)
        f_peptide[:forward_peptide_ids.shape[0]] = forward_peptide_ids
        batch_peptide_ids_forward.append(f_peptide)

    batch_forward_intensity = torch.from_numpy(np.stack(batch_forward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_peptide_ids_forward = torch.from_numpy(np.stack(batch_peptide_ids_forward))  # [batch_size, batch_max_seq_len]

    batch_backward_intensity = []
    batch_peptide_ids_backward = []
    for data in train_data_list:
        b_intensity = np.zeros((batch_max_seq_len, intensity_shape[0], intensity_shape[1], intensity_shape[2]),
                               np.float32)
        backward_intensity = np.stack(data.backward_candidate_intensity)
        b_intensity[:backward_intensity.shape[0], :, :, :] = backward_intensity
        batch_backward_intensity.append(b_intensity)

        b_peptide = np.zeros((batch_max_seq_len,), np.int64)
        backward_peptide_ids = np.array(data.peptide_ids_backward, np.int64)
        b_peptide[:backward_peptide_ids.shape[0]] = backward_peptide_ids
        batch_peptide_ids_backward.append(b_peptide)

    batch_backward_intensity = torch.from_numpy(
        np.stack(batch_backward_intensity))  # [batch_size, batch_max_seq_len, 26, 8, 10]
    batch_peptide_ids_backward = torch.from_numpy(np.stack(batch_peptide_ids_backward))  # [batch_size, batch_max_seq_len]

    return (spectrum_holder,
            batch_forward_intensity,
            batch_backward_intensity,
            batch_peptide_ids_forward,
            batch_peptide_ids_backward)

class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override the _get_spectrum method
    def _get_spectrum(self, feature: DIAFeature, input_spectrum_file_handle) -> DenovoData:
        """TODO(nh2tran): docstring."""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerIO: get_spectrum()")

        # parse and process spectrum
        (
            spectrum_holder,
            spectrum_original_forward,
            spectrum_original_backward,
            scan_list_middle,
            scan_list_original,
            ms1_profile,
        ) = self._parse_spectrum(
            feature.precursor_mz,
            feature.precursor_mass,
            feature.rt_mean,
            feature.scan_list,
            feature.ms1_list,
            input_spectrum_file_handle,
        )
        feature.scan_list_middle = scan_list_middle
        feature.scan_list = scan_list_original
        return DenovoData(spectrum_holder,
                          spectrum_original_forward,
                          spectrum_original_backward,
                          feature)

def denovo_collate_func(data_list: List[DenovoData]) -> BatchDenovoData:
    spectrum_holder = np.array([x.spectrum_holder for x in data_list])
    spectrum_holder = torch.from_numpy(spectrum_holder)
    spectrum_original_forward = [x.spectrum_original_forward for x in data_list]
    spectrum_original_backward = [x.spectrum_original_backward for x in data_list]
    dia_features = [x.dia_feature for x in data_list]
    return BatchDenovoData(spectrum_holder, spectrum_original_forward, spectrum_original_backward, dia_features)
