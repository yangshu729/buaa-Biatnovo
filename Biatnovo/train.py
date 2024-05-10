"""
This script handles the training process.
"""

import argparse
import time
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import deepnovo_config
import Biatnovo.Model.TrainingModel_indepedent as TM
from six.moves import xrange  # pylint: disable=redefined-builtin
from Biatnovo.DataProcessing import deepnovo_worker_io
from Biatnovo.DataProcessing.read import read_random_stack
from Biatnovo.Model.optim import ScheduledOptim


__author__ = "Si-yu Wu"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device_ids = [0, 1, 2, 3]
device_ids = [1]
# device_ids = [0, 1]
loss_count = 0


def cal_performance(pred_forward, pred_backward, gold_forward, gold_backward, trg_pad_idx, smoothing=False):
    """Apply label smoothing if needed"""
    loss = cal_loss(pred_forward, pred_backward, gold_forward, gold_backward, trg_pad_idx, smoothing)
    pred_forward = pred_forward.max(1)[1]
    gold_forward = gold_forward.contiguous().view(-1)
    pred_backward = pred_backward.max(1)[1]
    gold_backward = gold_backward.contiguous().view(-1)
    non_pad_mask_forward = gold_forward.ne(trg_pad_idx)
    non_pad_mask_backward = gold_backward.ne(trg_pad_idx)
    n_correct_forward = pred_forward.eq(gold_forward).masked_select(non_pad_mask_forward).sum().item()
    n_correct_backward = pred_backward.eq(gold_backward).masked_select(non_pad_mask_backward).sum().item()
    n_word = non_pad_mask_forward.sum().item() + non_pad_mask_backward.sum().item()
    return loss, n_correct_forward, n_correct_backward, n_word


def cal_loss(pred_forward, pred_backward, gold_forward, gold_backward, trg_pad_idx, smoothing=False):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    gold_forward = gold_forward.contiguous().view(-1)
    gold_backward = gold_backward.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred_forward.size(1)
        one_hot = torch.zeros_like(pred_forward).scatter(1, gold_forward.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred_forward, dim=1)

        non_pad_mask = gold_forward.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss_forward = F.cross_entropy(pred_forward, gold_forward, ignore_index=trg_pad_idx, reduction="sum")
        loss_backward = F.cross_entropy(pred_backward, gold_backward, ignore_index=trg_pad_idx, reduction="sum")
        loss = loss_forward + loss_backward
    return loss


def cal_performance_focal_loss(pred_forward, pred_backward, gold_forward, gold_backward, trg_pad_idx):
    # pred_forward.shape = (batchsize * (decoder_size - 1), num_classes) eg: [32 * 11, 26]
    zeros_forward = torch.zeros_like(gold_forward, dtype=gold_forward.dtype)
    ones_forward = torch.ones_like(gold_forward, dtype=gold_forward.dtype)
    gold_forward_weight = torch.where(gold_forward == 0, zeros_forward, ones_forward)
    zeros_backward = torch.zeros_like(gold_backward, dtype=gold_backward.dtype)
    ones_backward = torch.ones_like(gold_backward, dtype=gold_backward.dtype)
    gold_backward_weight = torch.where(gold_backward == 0, zeros_backward, ones_backward)
    loss_f = cal_folcal_loss(pred_forward, gold_forward) * gold_forward_weight
    loss_b = cal_folcal_loss(pred_backward, gold_backward) * gold_backward_weight
    loss_forward = torch.sum(loss_f)
    loss_backward = torch.sum(loss_b)
    loss = loss_forward + loss_backward
    pred_forward = pred_forward.max(1)[1] # (batchsize * (decoder_size - 1))
    gold_forward = gold_forward.contiguous().view(-1)
    pred_backward = pred_backward.max(1)[1]
    gold_backward = gold_backward.contiguous().view(-1)
    non_pad_mask_forward = gold_forward.ne(trg_pad_idx)
    non_pad_mask_backward = gold_backward.ne(trg_pad_idx)
    n_correct_forward = pred_forward.eq(gold_forward).masked_select(non_pad_mask_forward).sum().item()
    n_correct_backward = pred_backward.eq(gold_backward).masked_select(non_pad_mask_backward).sum().item()
    n_word = non_pad_mask_forward.sum().item() + non_pad_mask_backward.sum().item()
    return loss, n_correct_forward, n_correct_backward, n_word


def cal_folcal_loss(pred, gold, gamma=2):
    sigmoid_p = torch.sigmoid(pred)
    num_classes = pred.shape[-1]
    target_tensor = F.one_hot(gold, num_classes=num_classes)
    zeros = torch.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = torch.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = torch.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = -(pos_p_sub**gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - (
        neg_p_sub**gamma
    ) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    # (batchsize * (decoder_size - 1), 对每个样本的每个类别loss求和
    return torch.sum(per_entry_cross_ent, dim=-1)


def get_batch_2(index_list, data_set, bucket_id):
    """TODO(nh2tran): docstring."""

    batch_size = len(index_list)
    spectrum_holder_list = []
    candidate_intensity_lists_forward = []
    candidate_intensity_lists_backward = []
    decoder_inputs_forward = []
    decoder_inputs_backward = []
    for index in index_list:
        # Get a random entry of encoder and decoder inputs from data,
        (
            spectrum_holder,
            candidate_intensity_list_forward,
            candidate_intensity_list_backward,
            decoder_input_forward,
            decoder_input_backward,
        ) = data_set[bucket_id][index]
        if spectrum_holder is None:  # spectrum_holder is not provided if not use_lstm
            spectrum_holder = np.zeros(shape=(deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE), dtype=np.float32)
        spectrum_holder_list.append(spectrum_holder)
        candidate_intensity_lists_forward.append(candidate_intensity_list_forward)  # --> (batchsize, 12, 26, 40, 10)
        candidate_intensity_lists_backward.append(candidate_intensity_list_backward)
        decoder_inputs_forward.append(decoder_input_forward)  # --> (batchsize, 12)
        decoder_inputs_backward.append(decoder_input_backward)
    batch_spectrum_holder = np.array(spectrum_holder_list)
    batch_intensity_inputs_forward = []
    batch_intensity_inputs_backward = []
    batch_decoder_inputs_forward = []
    batch_decoder_inputs_backward = []
    batch_weights = []
    decoder_size = deepnovo_config._buckets[bucket_id]
    for length_idx in xrange(decoder_size):
        batch_intensity_inputs_forward.append(
            np.array(
                [candidate_intensity_lists_forward[batch_idx][length_idx] for batch_idx in xrange(batch_size)],
                dtype=np.float32,
            )
        )  # (12, batchsize, 26, 40, 10)
        batch_intensity_inputs_backward.append(
            np.array(
                [candidate_intensity_lists_backward[batch_idx][length_idx] for batch_idx in xrange(batch_size)],
                dtype=np.float32,
            )
        )
        batch_decoder_inputs_forward.append(
            np.array(
                [decoder_inputs_forward[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32
            )
        )  # (12, batchsize)
        batch_decoder_inputs_backward.append(
            np.array(
                [decoder_inputs_backward[batch_idx][length_idx] for batch_idx in xrange(batch_size)], dtype=np.int32
            )
        )
        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs_forward[batch_idx][length_idx + 1]
            # We set weight to 0 if the corresponding target is a PAD symbol.
            if (
                length_idx == decoder_size - 1
                or target == deepnovo_config.EOS_ID
                or target == deepnovo_config.GO_ID
                or target == deepnovo_config.PAD_ID
            ):
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return (
        batch_spectrum_holder,
        batch_intensity_inputs_forward,
        batch_intensity_inputs_backward,
        batch_decoder_inputs_forward,
        batch_decoder_inputs_backward,
        batch_weights,
    )


def create_model(opt, training_mode, device):
    """TODO(nh2tran): docstring."""
    print("".join(["="] * 80))  # section-separating line
    if os.path.exists(os.path.join(opt.train_dir, "translate.ckpt")):
        # load check_point
        print("loading checkpoint()")
        checkpoint = torch.load(os.path.join(opt.train_dir, "translate.ckpt"))
        model_opt = checkpoint["settings"]
        model = TM.TrainingModel(model_opt, training_mode)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print("create_model()")
        model = TM.TrainingModel(opt, training_mode).to(device)
        # gptmodel = load
        # model.decoder = gptmodel.decoder
        # model.tgr_word_project = gptmodel.tgtrwork_projection
        start_epoch = -1
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, start_epoch


def train_cycle(model, worker_io_train, feature_index_list_train, opt, optimizer, training_mode, step):
    """TODO(nh2tran): docstring."""
    worker_io_train.feature_count = dict.fromkeys(worker_io_train.feature_count, 0)
    train_set, _ = read_random_stack(
        worker_io_train, feature_index_list_train, deepnovo_config.train_stack_size, opt, training_mode, step
    )

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(deepnovo_config._buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print("train_bucket_sizes ", train_bucket_sizes)
    train_buckets_scale = [sum(train_bucket_sizes[: i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    train_current_spectra = [0 for b in xrange(len(deepnovo_config._buckets))]
    total_loss, n_word_total, n_word_correct_forward, n_word_correct_backward, n_word_correct_total = 0, 0, 0, 0, 0
    train_bucket_id = 0
    while True:
        if train_current_spectra[train_bucket_id] + deepnovo_config.batch_size > train_bucket_sizes[train_bucket_id]:
            train_bucket_id += 1
            if (
                train_bucket_id == len(deepnovo_config._buckets)
                or train_bucket_sizes[train_bucket_id] < deepnovo_config.batch_size
            ):
                print("train_current_spectra ", train_current_spectra)
                break
        index_list = range(
            train_current_spectra[train_bucket_id], train_current_spectra[train_bucket_id] + deepnovo_config.batch_size
        )
        (
            spectrum_holder, # (batchsize, neighbor_size, 150000)
            intensity_inputs_forward, # (12, batchsize, 26, 40, 10)
            intensity_inputs_backward, 
            decoder_inputs_forward, # (12, batchsize)
            decoder_inputs_backward,
            target_weights,
        ) = get_batch_2(index_list, train_set, train_bucket_id)
        # monitor the number of spectra that have been processed
        train_current_spectra[train_bucket_id] += deepnovo_config.batch_size
        decoder_size = deepnovo_config._buckets[train_bucket_id]
        model.train()
        optimizer.zero_grad()
        spectrum_holder = torch.from_numpy(spectrum_holder).cuda()
        # decoder_inputs_forward shape = (_buckets[train_bucket_id], batch_size)
        decoder_inputs_forward = torch.Tensor(decoder_inputs_forward).to(torch.int64).cuda()
        decoder_inputs_backward = torch.Tensor(decoder_inputs_backward).to(torch.int64).cuda()
        output_logits_forward, output_logits_backward = model(
            opt,
            spectrum_holder,
            intensity_inputs_forward,
            intensity_inputs_backward,
            decoder_inputs_forward[: decoder_size - 1],
            decoder_inputs_backward[: decoder_size - 1],
            True,
        )

        gold_forward = decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
        gold_backward = decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
        loss, n_correct_forward, n_correct_backward, n_word = cal_performance_focal_loss(
            output_logits_forward, output_logits_backward, gold_forward, gold_backward, 0
        )
        loss.backward()
        global loss_count
        loss_count += 1
        optimizer.step_and_update_lr()
        n_word_total += n_word  # forward + backward
        n_word_correct_forward += n_correct_forward
        n_word_correct_backward += n_correct_backward
        n_word_correct_total += n_correct_forward
        n_word_correct_total += n_correct_backward
        total_loss += loss.item()
    return total_loss, n_word_correct_forward, n_word_correct_backward, n_word_correct_total, n_word_total


def valid_test(model, opt, valid_set, valid_bucket_pos_id):
    total_loss, n_word_total, n_word_correct_forward, n_word_correct_backward, n_word_correct_total = 0, 0, 0, 0, 0
    for bucket_id in valid_bucket_pos_id:
        data_set_len = len(valid_set[bucket_id])
        data_set_index_list = range(data_set_len)
        data_set_index_chunk_list = [
            data_set_index_list[i : i + deepnovo_config.batch_size]
            for i in range(0, data_set_len, deepnovo_config.batch_size)
        ]
        for chunk in data_set_index_chunk_list:
            (
                spectrum_holder,
                intensity_inputs_forward,
                intensity_inputs_backward,
                decoder_inputs_forward,
                decoder_inputs_backward,
                target_weights,
            ) = get_batch_2(chunk, valid_set, bucket_id)
            decoder_size = deepnovo_config._buckets[bucket_id]
            spectrum_holder = torch.from_numpy(spectrum_holder).cuda()
            decoder_inputs_forward = torch.Tensor(decoder_inputs_forward).to(torch.int64).cuda()
            # (batchsize, seq len)
            decoder_inputs_backward = torch.Tensor(decoder_inputs_backward).to(torch.int64).cuda()
            output_logits_forward, output_logits_backward = model(
                opt,
                spectrum_holder,
                intensity_inputs_forward,
                intensity_inputs_backward,
                decoder_inputs_forward[: decoder_size - 1],
                decoder_inputs_backward[: decoder_size - 1],
                True,
            )
            gold_forward = decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1)
            gold_backward = decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            loss, n_correct_forward, n_correct_backward, n_word = cal_performance_focal_loss(
                output_logits_forward, output_logits_backward, gold_forward, gold_backward, 0
            )
            n_word_total += n_word  # forward + backward
            n_word_correct_forward += n_correct_forward
            n_word_correct_backward += n_correct_backward
            n_word_correct_total += n_correct_forward
            n_word_correct_total += n_correct_backward
            total_loss += loss.item()
    return n_word_correct_forward, n_word_correct_backward, n_word_correct_total, n_word_total, total_loss


def train(opt):
    print("".join(["="] * 80))  # section-separating line
    print("LoadFeature")
    ### input train and valid data
    worker_io_train = deepnovo_worker_io.WorkerIO(
        input_spectrum_file=opt.train_spectrum, input_feature_file=opt.train_feature
    )
    worker_io_valid = deepnovo_worker_io.WorkerIO(
        input_spectrum_file=opt.valid_spectrum, input_feature_file=opt.valid_feature
    )
    worker_io_train.open_input()
    worker_io_valid.open_input()
    worker_io_train.get_location()
    worker_io_valid.get_location()
    feature_index_list_train = worker_io_train.feature_index_list
    feature_index_list_valid = worker_io_valid.feature_index_list
    valid_set, valid_set_len = read_random_stack(
        worker_io_valid, feature_index_list_valid, deepnovo_config.valid_stack_size, opt
    )
    valid_bucket_len = [len(x) for x in valid_set]
    assert valid_set_len == sum(valid_bucket_len), "Error: valid_set_len"
    valid_bucket_pos_id = np.nonzero(valid_bucket_len)[0]
    ### log_file to record perplexity/accuracy during training
    train_log_file = opt.train_dir + "/train.log"
    valid_log_file = opt.train_dir + "/valid.log"
    print("Open train log_file: ", train_log_file)
    print("Open valid log_file: ", valid_log_file)
    with open(train_log_file, "w") as log_tf, open(valid_log_file, "w") as log_vf:
        log_tf.write("epoch,step,loss, accuracy_forward, accuracy_backward, accuracy_total\n")
        log_vf.write("epoch,valid_loss, accuracy_forward, accuracy_backward, accuracy_total\n")
    device = torch.device("cuda" if opt.cuda else "cpu")
    model, start_epoch = create_model(opt, True, device)
    checkpoint_path = os.path.join(opt.train_dir, "translate.ckpt")
    print("Model directory: ", checkpoint_path)
    if len(feature_index_list_train) % deepnovo_config.train_stack_size == 0:
        step_len = len(feature_index_list_train) // deepnovo_config.train_stack_size
    else:
        step_len = len(feature_index_list_train) // deepnovo_config.train_stack_size + 1
    print("Training loop")
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), opt.lr_mul, opt.d_model, opt.n_warmup_steps
    )
    start_time = time.time()
    valid_count = 0
    min_loss = sys.maxsize
    for epoch in range(start_epoch + 1, deepnovo_config.epoch_stop):
        (
            epoch_loss,
            epoch_word_correct_forward,
            epoch_word_correct_backward,
            epoch_word_correct_total,
            epoch_word_total,
        ) = (0, 0, 0, 0, 0)
        for step in range(step_len):
            (
                step_loss,
                step_word_correct_forward,
                step_word_correct_backward,
                step_word_correct_total,
                step_word_total,
            ) = train_cycle(
                model, worker_io_train, feature_index_list_train, opt, optimizer, training_mode=True, step=step
            )
            print(
                "epoch: ",
                str(epoch),
                "step :",
                str(step),
                " ",
                "step_loss :",
                str(step_loss / deepnovo_config.train_stack_size) + " " + "step word forward accuarcy:",
                str(step_word_correct_forward / (step_word_total / 2)) + " step word backward accuarcy:",
                str(step_word_correct_backward / (step_word_total / 2)) + " step word accuarcy:",
                str(step_word_correct_total / step_word_total),
            )
            epoch_loss += step_loss / deepnovo_config.train_stack_size
            epoch_word_correct_forward += step_word_correct_forward
            epoch_word_correct_backward += step_word_correct_backward
            epoch_word_correct_total += step_word_correct_forward
            epoch_word_correct_total += step_word_correct_backward
            epoch_word_total += step_word_total
            with open(train_log_file, "a") as log_tf:
                log_tf.write(
                    "{epoch},{step},{loss: 8.5f},{accu_f:3.3f},{accu_b:3.3f},{accu_t:3.3f}\n".format(
                        epoch=epoch,
                        step=step,
                        loss=step_loss / deepnovo_config.train_stack_size,
                        accu_f=100 * (step_word_correct_forward / (step_word_total / 2)),
                        accu_b=100 * (step_word_correct_backward / (step_word_total / 2)),
                        accu_t=100 * (step_word_correct_total / step_word_total),
                    )
                )
        print(
            "epoch :",
            str(epoch),
            " ",
            "epoch_loss :",
            str(epoch_loss / step_len) + " " + "epoch word forward accuarcy:",
            str(epoch_word_correct_forward / (epoch_word_total / 2)),
            " epoch word backward accuarcy:",
            str(epoch_word_correct_backward / (epoch_word_total / 2)),
            " epoch word accuarcy:",
            str(epoch_word_correct_total / epoch_word_total),
        )
        epoch_time = time.time() - start_time
        print("epoch_time:", epoch_time)
        (
            valid_word_correct_forward,
            valid_word_correct_backward,
            valid_word_correct_total,
            valid_word_total,
            valid_loss,
        ) = valid_test(model, opt, valid_set, valid_bucket_pos_id)
        print(
            "valid : " + "valid word forward accuarcy:",
            str(valid_word_correct_forward / (valid_word_total / 2)),
            " valid word backward accuarcy:",
            str(valid_word_correct_backward / (valid_word_total / 2)),
            " valid word accuarcy:",
            str(valid_word_correct_total / valid_word_total),
            " valid loss:",
            str(valid_loss / deepnovo_config.valid_stack_size),
        )
        with open(valid_log_file, "a") as log_vf:
            log_vf.write(
                "{epoch},{valid_lo:3.3f},{accu_f:3.3f},{accu_b:3.3f},{accu_t:3.3f}\n".format(
                    epoch=epoch,
                    valid_lo=100 * (valid_loss / deepnovo_config.valid_stack_size),
                    accu_f=100 * (valid_word_correct_forward / (valid_word_total / 2)),
                    accu_b=100 * (valid_word_correct_backward / (valid_word_total / 2)),
                    accu_t=100 * (valid_word_correct_total / valid_word_total),
                )
            )
        if valid_loss < min_loss:
            # save checkpoint
            valid_count = 0
            min_loss = min(min_loss, valid_loss)
            print("Current min loss:", min_loss)
            checkpoint = {"epoch": epoch, "settings": opt, "model": model.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            print("    - [Info] The checkpoint file has been updated.")
        else:
            valid_count += 1
            if valid_count > 5:
                break


def main():
    """
    Usage:
    python train.py --train --train_dir train_dir --train_spectrum --train_feature --valid_spectrum --valid_feature
     --use_intensity --use_lstm  --cuda
    """
    parser = argparse.ArgumentParser()
    # ==============================================================================
    # FLAGS (options) for train
    # ==============================================================================
    parser.add_argument("--train", action="store_true", default=False, help="Set to True for training.")
    parser.add_argument("--train_dir", type=str, default="train/", help="Training directory")
    parser.add_argument(
        "--train_spectrum", type=str, default="train_spectrum", help="Spectrum mgf file to train a new model."
    )
    parser.add_argument(
        "--train_feature", type=str, default="train_feature", help="Feature csv file to train a new model."
    )
    parser.add_argument(
        "--valid_spectrum", type=str, default="train_spectrum", help="Spectrum mgf file for validation during training."
    )
    parser.add_argument(
        "--valid_feature", type=str, default="train_feature", help="Feature csv file for validation during training."
    )
    parser.add_argument("--direction", type=int, default=2, help="Set to 0/1/2 for Forward/Backward/Bi-directional.")
    parser.add_argument(
        "--use_intensity", action="store_true", default=False, help="Set to True to use intensity-model."
    )
    parser.add_argument("--shared", action="store_true", default=False, help="Set to True to use shared weights.")
    parser.add_argument("--use_lstm", action="store_true", default=False, help="Set to True to use lstm-model.")
    parser.add_argument("--cuda", action="store_true", default=True, help="Set to True to use gpu.")
    parser.add_argument(
        "--lstm_kmer",
        action="store_true",
        default=False,
        help="Set to True to use lstm model on k-mers instead of full sequence.",
    )
    parser.add_argument("--beam_search", action="store_true", default=False, help="Set to True for beam search.")
    parser.add_argument(
        "--multiprocessor", type=int, default=1, help="Use multi processors to read data during training."
    )
    parser.add_argument("-d_inner_hid", type=int, default=2048)
    parser.add_argument("-d_k", type=int, default=64)
    parser.add_argument("-d_v", type=int, default=64)
    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-n_layers", type=int, default=6)
    parser.add_argument("-lr_mul", type=float, default=2.0)
    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-warmup", "--n_warmup_steps", type=int, default=4000)
    opt = parser.parse_args()
    """
    Make Train Dir
    """
    if not os.path.exists(opt.train_dir):
        os.makedirs(opt.train_dir)
    print(opt)

    """
    Train Model
    """
    train(opt)


if __name__ == "__main__":
    main()
