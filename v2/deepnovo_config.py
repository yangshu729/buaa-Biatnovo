# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import calendar

import time
# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="train")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--search_denovo", dest="search_denovo", action="store_true")
parser.add_argument("--test", dest="test", action="store_true")

parser.set_defaults(train=False)
parser.set_defaults(search_denovo=False)
parser.set_defaults(test=False)

args = parser.parse_args()

train_dir = args.train_dir

# ==============================================================================
# GLOBAL VARIABLES for VOCABULARY
# ==============================================================================


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = [
    "A",
    "R",
    "N",
    "N(Deamidation)",
    "D",
    # ~ 'C',
    "C(Carbamidomethylation)",
    "E",
    "Q",
    "Q(Deamidation)",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "M(Oxidation)",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)

header_seq = True

# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949
mass_AA = {
    "_PAD": 0.0,
    "_GO": mass_N_terminus - mass_H,
    "_EOS": mass_C_terminus + mass_H,
    "A": 71.03711,  # 0
    "R": 156.10111,  # 1
    "N": 114.04293,  # 2
    "N(Deamidation)": 115.02695,
    "D": 115.02694,  # 3
    # ~ 'C(Carbamidomethylation)': 103.00919, # 4
    "C(Carbamidomethylation)": 160.03065,  # C(+57.02)
    # ~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
    "E": 129.04259,  # 5
    "Q": 128.05858,  # 6
    "Q(Deamidation)": 129.0426,
    "G": 57.02146,  # 7
    "H": 137.05891,  # 8
    "I": 113.08406,  # 9
    "L": 113.08406,  # 10
    "K": 128.09496,  # 11
    "M": 131.04049,  # 12
    "M(Oxidation)": 147.0354,
    "F": 147.06841,  # 13
    "P": 97.05276,  # 14
    "S": 87.03203,  # 15
    "T": 101.04768,  # 16
    "W": 186.07931,  # 17
    "Y": 163.06333,  # 18
    "V": 99.06841,  # 19
}

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)
mass_AA_min = mass_AA["G"]  # 57.02146

# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================

# if change, need to re-compile cython_speedup << NO NEED
# SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
# ~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
# ~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da
SPECTRUM_RESOLUTION = 50  # bins for 1.0 Da = precision 0.02 Da
# ~ SPECTRUM_RESOLUTION = 100 # bins for 1.0 Da = precision 0.01 Da
print("SPECTRUM_RESOLUTION ", SPECTRUM_RESOLUTION)

# if change, need to re-compile cython_speedup << NO NEED
WINDOW_SIZE = 10  # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

MZ_MAX = 3000.0
MZ_SIZE = int(MZ_MAX * SPECTRUM_RESOLUTION)  # 30k
KNAPSACK_AA_RESOLUTION = 10000  # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION))  # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100  # 0.01 Da
num_position = 0
PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN)
MAX_LEN = 30
print("MAX_LEN ", MAX_LEN)

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [12, 22, 32]
print("_buckets ", _buckets)


beam_size = 5
print("beam_size ", beam_size)
# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================


num_ion = 8  # 2
print("num_ion ", num_ion)

l2_weight = 0.0
print("l2_weight ", l2_weight)

# embedding_size = 512
embedding_size = 256
print("embedding_size ", embedding_size)


dropout_keep = {
  "conv" : 0.25,
  "dense" : 0.5,
  "transformer": 0.2
}

print("dropout_keep ", dropout_keep)


batch_size = 128
print("batch_size ", batch_size)

batch_size_predict = 32
print("batch_size_predict: ", batch_size_predict)

num_workers = 6
print("num_workers ", num_workers)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)

# DIA model parameters
neighbor_size = 5  # allow up to ? spectra, including the main spectrum
dia_window = 20.0  # the window size of MS2 scan in Dalton
focal_loss = True


# ==============================================================================
# DB SEARCH PARAMETERS
# ==============================================================================


data_format = "mgf"
cleavage_rule = "trypsin"
num_missed_cleavage = 2
fixed_mod_list = ["C"]
var_mod_list = ["N", "Q", "M"]
num_mod = 3
precursor_mass_tolerance = 0.01  # Da
precursor_mass_ppm = 15.0 / 1000000  # ppm (20 better) # instead of absolute 0.01 Da
topk_output = 1


# ==============================================================================
# INPUT/OUTPUT FILES
# ==============================================================================
input_feature_file_train = "/root/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/train_dataset_unique.csv"
input_spectrum_file_train = "/root/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/training.spectrum.mgf"
input_spectrum_file_valid = "/root/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/training.spectrum.mgf"
input_feature_file_valid = "/root/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/valid_dataset_unique.csv"
denovo_input_feature_file = "/root/biatnovo/DeepNovo-DIA/oc/oc_test.feature.csv"
denovo_input_spectrum_file = "/root/biatnovo/DeepNovo-DIA/oc/oc_test.spectrum.mgf"
# input_spectrum_file_test = "/root/biatnovo/deenovov2/spectrum.mgf"
# input_feature_file_test = "ABRF_DDA/features.csv.identified.test.nodup"

# pre-built knapsack matrix
knapsack_file = "knapsack.npy"

denovo_output_file = denovo_input_feature_file + str(calendar.timegm(time.gmtime())) + ".deepnovo_denovo"

# ==============================================================================
# feature file column format
# ==============================================================================

col_feature_id = "spec_group_id"
col_precursor_mz = "m/z"
col_precursor_charge = "z"
col_rt_mean = "rt_mean"
col_raw_sequence = "seq"
col_scan_list = "scans"
col_ms1_list = "profile"
col_feature_area = "feature area"
# # predicted file column format
# pcol_feature_id = 0
# pcol_feature_area = 1
# pcol_sequence = 2
# pcol_score = 3
# pcol_position_score = 4
# pcol_precursor_mz = 5
# pcol_precursor_charge = 6
# pcol_protein_id = 7
# pcol_scan_list_middle = 8
# pcol_scan_list_original = 9
# pcol_score_max = 10
# pcol_score_max_unlabeled = 9

cuda = True
lr_mul = 2.0  # 0.5
d_model = 256  # 256
n_warmup_steps = 4000
num_epoch = 30
steps_per_validation = 100
early_stop = 10

print("aaaaaa")