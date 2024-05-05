import datetime
import os
import time
import torch
import torch.optim as optim
from Biatnovo.Model.optim import ScheduledOptim
from test_accuracy import cal_dia_focal_loss
from v2.data_reader import *
from v2 import deepnovo_config
import logging

from v2.model import DeepNovoAttion
from v2.test_accuracy import test_logit_batch_2

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model(dropout_keep, training_mode):
    """TODO(nh2tran): docstring."""
    print("".join(["="] * 80))  # section-separating line
    if os.path.exists(os.path.join(deepnovo_config.train_dir, "translate.ckpt")):
        # load check_point
        print("loading checkpoint()")
        checkpoint = torch.load(os.path.join(deepnovo_config.train_dir, "translate.ckpt"))
        model = DeepNovoAttion(dropout_keep)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print("create_model()")
        model = DeepNovoAttion(dropout_keep).to(device)
        # gptmodel = load
        # model.decoder = gptmodel.decoder
        # model.tgr_word_project = gptmodel.tgtrwork_projection
        start_epoch = -1
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model, start_epoch


def validation(model, valid_loader, data_set_len) -> float:
    avg_accuracy_AA = 0
    avg_accuracy_peptide = 0
    avg_loss = 0
    avg_len_AA = 0
    avg_accuracy_len = 0
    with torch.no_grad():
        for data in valid_loader:
            (spectrum_holder, 
             batch_intensity_inputs_forward, # [batch_size, batch_max_seq_len, 26, 8 * 5, 10]
             batch_intensity_inputs_backward, 
             batch_decoder_inputs_forward,  # [batch_size, batch_max_seq_len]
             batch_decoder_inputs_backward) = data
            batch_size = spectrum_holder.size(0)

            # move to device
            spectrum_holder = spectrum_holder.to(device)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.to(device)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.to(device)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.to(device)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.to(device)

            # (batchsize, batch_max_seq_len, 26, 40, 10) -> (batch_max_seq_len, batchsize, 26, 40, 10)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.permute(1, 0, 2, 3, 4)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.permute(1, 0, 2, 3, 4)
            # (batch_max_seq_len, batchsize)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.permute(1, 0)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.permute(1, 0)

            # (batchsize x seq_len, 26))
            output_logits_forward, output_logits_backward = model(
                spectrum_holder,  # (batchsize, neighbor_size, 150000)
                batch_intensity_inputs_forward, # (seq_len, batchsize, 26, 40, 10)
                batch_intensity_inputs_backward, # (seq_len, batchsize, 26, 40, 10)
                # batch_decoder_inputs_forward eg: [[1, 3, 5, 6, ..., 2]]
                batch_decoder_inputs_forward[:- 1], # (seq_len - 1, batchsize)
                batch_decoder_inputs_backward[:- 1],
            )

            gold_forward = batch_decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
            gold_backward = batch_decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            output_logits_forward_trans = output_logits_forward.view(-1, output_logits_forward.size(2))
            output_logits_backward_trans = output_logits_backward.view(-1, output_logits_backward.size(2))
            output_logits_forward = output_logits_forward.transpose(0, 1) 
            output_logits_backward = output_logits_backward.transpose(0, 1) # (seq_len, batchsize, 26)
            loss = cal_dia_focal_loss(output_logits_forward_trans, output_logits_backward_trans, gold_forward, gold_backward, 0)
            # gold_forward = batch_decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
            # gold_backward = batch_decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            # loss = cal_dia_focal_loss(output_logits_forward, output_logits_backward, gold_forward, gold_backward, 0)
             
            (batch_accuracy_AA,
             batch_len_AA,
             num_exact_match,
             num_len_match) = test_logit_batch_2(batch_decoder_inputs_forward,
                                    batch_decoder_inputs_backward,
                                    output_logits_forward,
                                    output_logits_backward)
            avg_loss += loss 

            avg_accuracy_AA += batch_accuracy_AA
            avg_len_AA += batch_len_AA
            avg_accuracy_peptide += num_exact_match
            avg_accuracy_len += num_len_match

    avg_loss /= data_set_len
    avg_accuracy_AA /= avg_len_AA
    avg_accuracy_peptide /= data_set_len
    
    return avg_loss.item(), avg_accuracy_AA, avg_accuracy_peptide

def train():
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train,
                                     deepnovo_config.input_spectrum_file_train)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features / deepnovo_config.batch_size)
    logger.info(f"{steps_per_epoch} steps per epoch")
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=True,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    valid_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_valid,
                                     deepnovo_config.input_spectrum_file_valid)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=False,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    
    # Get current date and time
    current_time = datetime.datetime.now()

    # Format the datetime string
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the log file name with the current date and time
    log_file = f"{deepnovo_config.train_dir}/log_file_caption_2dir_{formatted_time}.tab"

    logger.info(f"Open log_file: {log_file}")

    with open(log_file, 'a') as log_file_handle:
        print("epoch\tstep\tloss\tlast_accuracy_AA\tlast_accuracy_peptide\tvalid_loss\tvalid_accuracy_AA\tvalid_accuracy_peptide\n",
            file=log_file_handle, end="")

    model, start_epoch = create_model(deepnovo_config.dropout_keep, True, device)
    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                    deepnovo_config.lr_mul, deepnovo_config.d_model, deepnovo_config.n_warmup_steps
    )
    checkpoint_path = os.path.join(deepnovo_config.train_dir, "translate.ckpt")
    
    best_valid_loss = float("inf")
    # train loop
    best_epoch = None
    best_step = None
    start_time = time.time()
    recent_losses = []  # 保存最近deepnovo_config.steps_per_validation

    for epoch in range(deepnovo_config.num_epoch):
        # learning rate schedule
        # adjust_learning_rate(optimizer, epoch)
        for i, data in enumerate(train_data_loader):
            optimizer.zero_grad() # clear previous gradients
            # (batchsize, neighbor_size, 150000)
            (spectrum_holder, 
             batch_intensity_inputs_forward, # [batch_size, batch_max_seq_len, 26, 8 * 5, 10]
             batch_intensity_inputs_backward, 
             batch_decoder_inputs_forward,  # [batch_size, batch_max_seq_len]
             batch_decoder_inputs_backward) = data
            batchsize = spectrum_holder.size(0)

            # move to device
            spectrum_holder = spectrum_holder.to(device)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.to(device)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.to(device)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.to(device)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.to(device)

            # (eq_len, batchsize, 26, 40, 10)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.permute(1, 0, 2, 3, 4)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.permute(1, 0, 2, 3, 4)
            # (seq_len, batchsize)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.permute(1, 0)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.permute(1, 0)

            ## (batchsize , seq len, 26)
            output_logits_forward, output_logits_backward = model(
                spectrum_holder,  # (batchsize, neighbor_size, 150000)
                batch_intensity_inputs_forward, # (seq_len, batchsize, 26, 40, 10)
                batch_intensity_inputs_backward, # (seq_len, batchsize, 26, 40, 10)
                batch_decoder_inputs_forward[: -1], # (seq_len, batchsize)
                batch_decoder_inputs_backward[: -1],
            )
            gold_forward = batch_decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
            gold_backward = batch_decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            output_logits_forward_trans = output_logits_forward.view(-1, output_logits_forward.size(2))
            output_logits_backward_trans = output_logits_backward.view(-1, output_logits_backward.size(2))
            output_logits_forward = output_logits_forward.transpose(0, 1) 
            output_logits_backward = output_logits_backward.transpose(0, 1) # (seq_len, batchsize, 26)
            loss = cal_dia_focal_loss(output_logits_forward_trans, output_logits_backward_trans, gold_forward, gold_backward, 0)
            loss.backward()
            optimizer.step_and_update_lr()
            # 更新最近的loss列表
            if len(recent_losses) < deepnovo_config.steps_per_validation:
                recent_losses.append(loss.item() / batchsize)
            else:
                recent_losses.pop(0)
                recent_losses.append(loss.item() / batchsize)
            if (i + 1) % deepnovo_config.steps_per_validation == 0:
                duration = time.time() - start_time
                step_time = duration / deepnovo_config.steps_per_validation
                # loss are averaged over last  steps_per_validation
                avg_loss = sum(recent_losses) / len(recent_losses)
                
                # accuracy is last batch
                (batch_accuracy_AA,
                 batch_len_AA,
                 num_exact_match,
                 _) = test_logit_batch_2(batch_decoder_inputs_forward, # (seq_len, batchsize)
                                        batch_decoder_inputs_backward,
                                        output_logits_forward,
                                        output_logits_backward)
                # print update of accuracy, time, loss
                accuracy_AA = batch_accuracy_AA / batch_len_AA
                accuracy_peptide = num_exact_match / deepnovo_config.batch_size

                with open(log_file, 'a') as log_file_handle:
                    # Print the formatted data directly to the file
                    print("%d\t%d\t%.4f\t%.4f\t%.4f\t" %
                        (epoch,
                        i + 1,
                        avg_loss,
                        accuracy_AA,
                        accuracy_peptide),
                        file=log_file_handle,
                        end="")
                # validation
                model.eval()
                validaion_begin_time = time.time()
                validation_loss, accuracy_AA, accuracy_peptide  = validation(model, valid_data_loader, len(valid_set))
                validation_duration = time.time() - validaion_begin_time
                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train loss: {avg_loss}\tstep time: {step_time} "
                            f"validation loss: {validation_loss}")
                logger.info(f"validation cost: {validation_duration} seconds")
                with open(log_file, 'a') as log_file_handle:
                    print("%.4f \t %.4f \t %.4f\n" % (validation_loss, accuracy_AA, accuracy_peptide),
                    file=log_file_handle,
                    end="")
                model.train()
                start_time = time.time()
                if validation_loss < best_valid_loss:
                    no_update_count = 0
                    best_valid_loss = validation_loss
                    logger.info(f"best valid loss achieved at epoch {epoch} step {i}")
                    checkpoint = {"epoch": epoch,  "model": model.state_dict()}
                    torch.save(checkpoint, checkpoint_path)
                    best_epoch = epoch
                    best_step = i
                else:
                    no_update_count += 1
                    if no_update_count >= deepnovo_config.early_stop:
                        logger.info(f"early stop at epoch {epoch} step {i}")
                        break
        
        if no_update_count >= deepnovo_config.early_stop:
            break

    logger.info(f"best model at epoch {best_epoch} step {best_step}")