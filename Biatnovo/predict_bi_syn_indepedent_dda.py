"""
This script handles the training process.
"""

import argparse
import os
import torch
import Model.TrainingModel as TM
import Model.TrainingModel_indepedent as TM_Indepedent
from DataProcessing import deepnovo_worker_io
from DataProcessing import deepnovo_worker_denovo

__author__ = "Si-yu Wu"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_model(opt, device):
    checkpoint = torch.load(opt.model, map_location=device)
    model = TM.TrainingModel_DDA(opt, False).to(device)
    model.load_state_dict(checkpoint["model"])
    print("Trained model state loaded.")
    return model


def load_model_indepedent(opt, device):
    checkpoint_indepedent = torch.load(opt.model_indepedent, map_location=device)
    model_inde = TM_Indepedent.TrainingModel_DDA(opt, False).to(device)
    model_inde.load_state_dict(checkpoint_indepedent["model"])
    print("Trained model state loaded.")
    return model_inde


def predict(opt, model, model_ind):
    worker_io = deepnovo_worker_io.WorkerIO(
        input_spectrum_file=opt.predict_spectrum,
        input_feature_file=opt.predict_feature,
        output_file=opt.predict_dir,
        type="DDA",
    )
    # change batchsize
    worker_io.predict(type="DDA")
    worker_denovo = deepnovo_worker_denovo.WorkerDenovo(type="DDA")
    # worker_denovo.search_denovo_bi_SB_indepedent(model, model_ind, worker_io, opt)
    worker_denovo.search_denovo_bi_SB_indepedent_all(model, model_ind, worker_io, opt, type="DDA")


def main():
    """
    Usage:
    python train.py --train --train_dir train_dir --train_spectrum --train_feature --valid_spectrum --valid_feature
     --use_intensity --use_lstm  --cuda
    """

    parser = argparse.ArgumentParser()
    # ==============================================================================
    # FLAGS (options) for predict
    # ==============================================================================
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="Set to True to do a denovo search.",
    )

    parser.add_argument("--model", type=str, default="translate.ckpt", help="Training model checkpoint")
    parser.add_argument(
        "--model_indepedent",
        type=str,
        default="translate.ckpt",
        help="Training model checkpoint indepedent",
    )
    parser.add_argument("--predict_dir", type=str, default="predict/", help="Predicting directory")
    parser.add_argument(
        "--predict_spectrum",
        type=str,
        default="predict_spectrum",
        help="Spectrum mgf file to perform de novo sequencing.",
    )
    parser.add_argument(
        "--predict_feature",
        type=str,
        default="predict_feature",
        help="Feature csv file to perform de novo sequencing.",
    )
    parser.add_argument(
        "--direction",
        type=int,
        default=2,
        help="Set to 0/1/2 for Forward/Backward/Bi-directional.",
    )
    parser.add_argument(
        "--use_intensity",
        action="store_true",
        default=False,
        help="Set to True to use intensity-model.",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        default=False,
        help="Set to True to use shared weights.",
    )
    parser.add_argument(
        "--use_lstm",
        action="store_true",
        default=False,
        help="Set to True to use lstm-model.",
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="Set to True to use gpu.")
    parser.add_argument(
        "--lstm_kmer",
        action="store_true",
        default=False,
        help="Set to True to use lstm model on k-mers instead of full sequence.",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        default=False,
        help="Set to True for beam search.",
    )
    parser.add_argument(
        "--multiprocessor",
        type=int,
        default=1,
        help="Use multi processors to read data during training.",
    )

    opt = parser.parse_args()

    """
    Make Predict Dir
    """
    # if not os.path.exists(opt.predict_dir):
    #     os.makedirs(opt.predict_dir)
    print(opt)

    """
    Predict
    """
    device = torch.device("cuda" if opt.cuda else "cpu")
    # load model
    model = load_model(opt, device)
    model_ind = load_model_indepedent(opt, device)
    predict(opt, model, model_ind)


if __name__ == "__main__":
    main()
