import datetime
import math
import os
import time
import torch
import torch.optim as optim
from Biatnovo.Model.optim import ScheduledOptim
from test_accuracy import cal_dia_focal_loss
from v2.data_reader import *
from v2 import deepnovo_config
import logging

from v2.model import DeepNovoAttion, SpectrumCNN2
from v2.test_accuracy import cal_sb_dia_focal_loss, test_logit_batch_2

forward_model_save_name = 'forward_deepnovo.pth'
backward_model_save_name = 'backward_deepnovo.pth'
spectrum_cnn_save_name = 'spectrum_cnn.pth_1'
sbatt_model_save_name = 'sbatt_deepnovo.pth_1'
optimizer_save_name = 'optimizer.pth_1'

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_half_mask(seq_tensor, pad_val=0, mask_front=False):
    print("generate_half_mask")
    """
    Generates a mask for either the first half or the second half of the actual sequence length.
    Returns:
    torch.Tensor: A mask tensor of the same shape as `seq_tensor` with 0s in the masked positions and 1s elsewhere.
    """
    # Get the lengths of the sequences (ignoring padding)
    real_lengths = (seq_tensor != pad_val).sum(dim=1)
    
    # Compute half lengths
    half_lengths = (real_lengths / 2).ceil().long()
    
    # Create the base mask matrix where each row is [0, 1, 2, ..., seq_length-1]
    batch_size, seq_length = seq_tensor.size()
    mask = torch.arange(seq_length, device=seq_tensor.device).expand(batch_size, seq_length)
    
    if mask_front:
        # Mask the front half: keep positions less than half length as 0, others as 1
        mask = (mask >= half_lengths.unsqueeze(1)).long()
    else:
        # Mask the back half: keep positions less than half length as 1, others as 0
        mask = (mask < half_lengths.unsqueeze(1)).long()
    return mask

def create_model(dropout_keep):
    """TODO(nh2tran): docstring."""
    print("".join(["="] * 80))  # section-separating line
    spectrum_cnn = SpectrumCNN2().to(device)
    forward_deepnovo = DeepNovoAttion(dropout_keep)
    backward_deepnovo = DeepNovoAttion(dropout_keep)
    if os.path.exists(os.path.join(deepnovo_config.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(deepnovo_config.train_dir, backward_model_save_name))
        # load check_point
        print("loading pretrained model")
        assert os.path.exists(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name))
        spectrum_cnn.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name),
                                                    map_location=device))
        forward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, forward_model_save_name),
                                                    map_location=device))
        backward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, backward_model_save_name),
                                                     map_location=device))
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print("create_model()")
    backward_deepnovo.transformer.tgt_tok_emb.embedding.weight = forward_deepnovo.transformer.tgt_tok_emb.embedding.weight
    forward_deepnovo = forward_deepnovo.to(device)
    backward_deepnovo = backward_deepnovo.to(device)
    
    return forward_deepnovo, backward_deepnovo, spectrum_cnn

def create_sb_model(dropout_keep):
    """TODO(nh2tran): docstring."""
    print("".join(["="] * 80))  # section-separating line
    spectrum_cnn = SpectrumCNN2().to(device)
    sbatt_model = DeepNovoAttion(dropout_keep)
    if os.path.exists(os.path.join(deepnovo_config.train_dir, sbatt_model_save_name)):
        # load check_point
        logger.info("loading pretrained model")
        assert os.path.exists(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name))
        spectrum_cnn.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name),
                                                    map_location=device))

        sbatt_model.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, sbatt_model_save_name), map_location=device))
    else:
        logger.info("create_model()")
    sbatt_model = sbatt_model.to(device)
    # import torch.onnx
    # torch.onnx.export(sbatt_model, (torch.randn(1, 1, 256).to(device), torch.randn(1, 1, 26, 40, 10).to(device), 
    #                                 torch.randn(1, 1, 26, 40, 10).to(device), torch.randint(1, 26, (1, 1)).to(device),
    #                                 torch.randint(1, 26, (1, 1)).to(device)), "sbatt_model.onnx") 
    
    return sbatt_model, spectrum_cnn



def load_optimizer(optimizer: optim.Optimizer):
    if os.path.exists(os.path.join(deepnovo_config.train_dir, optimizer_save_name)):
        optimizer.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, optimizer_save_name)))
        return optimizer
    return optimizer


def save_model(spectrum_cnn, optimizer, forward_deepnovo = None, backward_deepnovo = None, sbatt_model = None, epoch = None):
    # if deepnovo_config.is_sb:
    torch.save(sbatt_model.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                         sbatt_model_save_name + f"_{epoch}"))
    # else:
    #     torch.save(forward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
    #                                                         forward_model_save_name))
    #     torch.save(backward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
    #                                                             backward_model_save_name))
    
    torch.save(spectrum_cnn.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                       spectrum_cnn_save_name + f"_{epoch}"))
    torch.save(optimizer.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                    optimizer_save_name + f"_{epoch}"))

def validation(sepctrum_cnn, valid_loader, data_set_len,
               forward_deepnovo = None,
               backward_deepnovo = None,
               sbatt_model = None) -> float:
    avg_accuracy_AA = 0
    avg_accuracy_peptide = 0
    avg_loss = 0
    avg_len_AA = 0
    avg_accuracy_len = 0
    with torch.no_grad():
        for data in valid_loader:
            if deepnovo_config.with_extra_predicted_training_sequence:
                (spectrum_holder,
                 batch_intensity_inputs_forward, # [batch_size, batch_max_seq_len, 26, 8 * 5, 10]
                 batch_intensity_inputs_backward,
                 batch_decoder_inputs_forward,  # [batch_size, batch_max_seq_len]
                 batch_decoder_inputs_backward,
                 batch_decoder_inputs_predicted_f,
                 batch_decoder_inputs_predicted_b) = data
            else:
                (spectrum_holder,
                batch_intensity_inputs_forward, # [batch_size, batch_max_seq_len, 26, 8 * 5, 10]
                batch_intensity_inputs_backward,
                batch_decoder_inputs_forward,  # [batch_size, batch_max_seq_len]
                batch_decoder_inputs_backward) = data
            batch_size = spectrum_holder.size(0)

            # move to device
            spectrum_holder = spectrum_holder.to(device)
            spectrum_cnn_outputs = sepctrum_cnn(spectrum_holder).permute(1, 0, 2)  # (batchsize, 1, 256)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.to(device)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.to(device)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.to(device)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.to(device)
            if deepnovo_config.with_extra_predicted_training_sequence:
                batch_decoder_inputs_predicted_f = batch_decoder_inputs_predicted_f.to(device)
                batch_decoder_inputs_predicted_b = batch_decoder_inputs_predicted_b.to(device)


            # (batchsize, batch_max_seq_len, 26, 40, 10) -> (batch_max_seq_len, batchsize, 26, 40, 10)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.permute(1, 0, 2, 3, 4)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.permute(1, 0, 2, 3, 4)
            # (batch_max_seq_len, batchsize)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.permute(1, 0)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.permute(1, 0)

            # if deepnovo_config.is_sb:
            if deepnovo_config.with_extra_predicted_training_sequence:
                output_logits_forward, output_logits_backward = sbatt_model(spectrum_cnn_outputs, 
                                                                            batch_intensity_inputs_forward, 
                                                                            batch_intensity_inputs_backward,
                                                                            batch_decoder_inputs_forward[:-1],
                                                                            batch_decoder_inputs_backward[:-1],
                                                                            batch_decoder_inputs_predicted_f[:, :-1],
                                                                            batch_decoder_inputs_predicted_b[:, :-1])
            else:
                output_logits_forward, output_logits_backward = sbatt_model(spectrum_cnn_outputs, 
                                                                            batch_intensity_inputs_forward, 
                                                                            batch_intensity_inputs_backward,
                                                                            batch_decoder_inputs_forward[:-1],
                                                                            batch_decoder_inputs_backward[:-1])
            # else:
            #     output_logits_forward = forward_deepnovo(spectrum_cnn_outputs, batch_intensity_inputs_forward, batch_decoder_inputs_forward[:-1])
            #     output_logits_backward = backward_deepnovo(spectrum_cnn_outputs, batch_intensity_inputs_backward, batch_decoder_inputs_backward[:-1])

            gold_forward = batch_decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
            gold_backward = batch_decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            output_logits_forward_trans = output_logits_forward.reshape(-1, output_logits_forward.size(2))
            output_logits_backward_trans = output_logits_backward.reshape(-1, output_logits_backward.size(2))
            output_logits_forward = output_logits_forward.transpose(0, 1)
            output_logits_backward = output_logits_backward.transpose(0, 1) # (seq_len, batchsize, 26)
            loss = cal_dia_focal_loss(output_logits_forward_trans, output_logits_backward_trans, gold_forward, gold_backward, batch_size)
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
            avg_loss += loss * batch_size

            avg_accuracy_AA += batch_accuracy_AA
            avg_len_AA += batch_len_AA
            avg_accuracy_peptide += num_exact_match
            avg_accuracy_len += num_len_match

    avg_loss /= data_set_len
    avg_accuracy_AA /= avg_len_AA
    avg_accuracy_peptide /= data_set_len
    eval_ppx = math.exp(avg_loss) if avg_loss < 300 else float('inf')

    return eval_ppx, avg_accuracy_AA, avg_accuracy_peptide

def train():
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train,
                                     deepnovo_config.input_spectrum_file_train,
                                     deepnovo_config.extra_predicted_training_sequence_file)
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
        print("epoch\tstep\tperplexity\tlast_accuracy_AA\tlast_accuracy_peptide\tvalid_perplexity\tvalid_accuracy_AA\tvalid_accuracy_peptide\n",
            file=log_file_handle, end="")
    
    sbatt_model, spectrum_cnn = create_sb_model(deepnovo_config.dropout_keep)
    all_params = list(sbatt_model.parameters()) + list(spectrum_cnn.parameters())
    # else:
    #     forward_deepnovo, backward_deepnovo, spectrum_cnn = create_model(deepnovo_config.dropout_keep)

    #     all_params = list(forward_deepnovo.parameters()) + list(backward_deepnovo.parameters()) + \
    #                 list(spectrum_cnn.parameters())
    
    optimizer = ScheduledOptim(
        optim.Adam(all_params, betas=(0.9, 0.98), eps=1e-09),
                    deepnovo_config.lr_mul, deepnovo_config.d_model, deepnovo_config.n_warmup_steps
    )
    optimizer = load_optimizer(optimizer)

    best_valid_loss = float("inf")
    # train loop
    best_epoch = None
    best_step = None
    start_time = time.time()
    recent_losses = []  # 保存最近deepnovo_config.steps_per_validation

    for epoch in range(deepnovo_config.num_epoch):
        # learning rate schedule
        # adjust_learning_rate(optimizer, epoch)
        new_loss = 0.0
        for i, data in enumerate(train_data_loader):
            logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}")
            optimizer.zero_grad() # clear previous gradients
            # (batchsize, neighbor_size, 150000)
            if deepnovo_config.with_extra_predicted_training_sequence:
                (spectrum_holder,
                 batch_intensity_inputs_forward, # [batch_size, batch_max_seq_len, 26, 8 * 5, 10]
                 batch_intensity_inputs_backward,
                 batch_decoder_inputs_forward,  # [batch_size, batch_max_seq_len]
                 batch_decoder_inputs_backward,
                 batch_decoder_inputs_predicted_f,
                 batch_decoder_inputs_predicted_b) = data
            else:
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
            if deepnovo_config.with_extra_predicted_training_sequence:
                batch_decoder_inputs_predicted_f = batch_decoder_inputs_predicted_f.to(device)
                batch_decoder_inputs_predicted_b = batch_decoder_inputs_predicted_b.to(device)

            # (seq_len, batchsize, 26, 40, 10)
            batch_intensity_inputs_forward = batch_intensity_inputs_forward.permute(1, 0, 2, 3, 4)
            batch_intensity_inputs_backward = batch_intensity_inputs_backward.permute(1, 0, 2, 3, 4)
            # (seq_len, batchsize)
            batch_decoder_inputs_forward = batch_decoder_inputs_forward.permute(1, 0)
            batch_decoder_inputs_backward = batch_decoder_inputs_backward.permute(1, 0)

            spectrum_cnn_outputs = spectrum_cnn(spectrum_holder).permute(1, 0, 2)  # (batchsize, 1, 256)

            if deepnovo_config.with_extra_predicted_training_sequence:
                output_logits_forward, output_logits_backward = sbatt_model(spectrum_cnn_outputs, 
                                                                                batch_intensity_inputs_forward, 
                                                                                batch_intensity_inputs_backward,
                                                                                batch_decoder_inputs_forward[: -1],
                                                                                batch_decoder_inputs_backward[: -1],
                                                                                batch_decoder_inputs_predicted_f[:, :-1],
                                                                                batch_decoder_inputs_predicted_b[:, :-1])
            else:
                output_logits_forward, output_logits_backward = sbatt_model(spectrum_cnn_outputs, 
                                                                                batch_intensity_inputs_forward, 
                                                                                batch_intensity_inputs_backward,
                                                                                batch_decoder_inputs_forward[: -1],
                                                                                batch_decoder_inputs_backward[: -1])
           
            # ## (batchsize , seq len, 26)
            # output_logits_forward, output_logits_backward = model(
            #     spectrum_holder,  # (batchsize, neighbor_size, 150000)
            #     batch_intensity_inputs_forward, # (seq_len, batchsize, 26, 40, 10)
            #     batch_intensity_inputs_backward, # (seq_len, batchsize, 26, 40, 10)
            #     batch_decoder_inputs_forward[: -1], # (seq_len, batchsize)
            #     batch_decoder_inputs_backward[: -1],
            # )
           
            gold_forward = batch_decoder_inputs_forward[1:].permute(1, 0).contiguous().view(-1) # (batchsize * (decoder_size - 1))
            gold_backward = batch_decoder_inputs_backward[1:].permute(1, 0).contiguous().view(-1)
            output_logits_forward_trans = output_logits_forward.reshape(-1, output_logits_forward.size(2))
            output_logits_backward_trans = output_logits_backward.reshape(-1, output_logits_backward.size(2))
            output_logits_forward = output_logits_forward.transpose(0, 1)
            output_logits_backward = output_logits_backward.transpose(0, 1) # (seq_len, batchsize, 26)
            loss = cal_dia_focal_loss(output_logits_forward_trans, output_logits_backward_trans, gold_forward, gold_backward, batchsize)
            loss.backward()
            
            new_loss += loss.item() / deepnovo_config.steps_per_validation
            optimizer.step_and_update_lr()

            if (i + 1) % deepnovo_config.steps_per_validation == 0:
                duration = time.time() - start_time
                step_time = duration / deepnovo_config.steps_per_validation
                # loss are averaged over last  steps_per_validation
                perplexity = math.exp(new_loss) if new_loss < 300 else float('inf')
                new_loss = 0.0

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
                        perplexity,
                        accuracy_AA,
                        accuracy_peptide),
                        file=log_file_handle,
                        end="")
                    # print("%d\t%d\t%.4f\t%.4f\t%.4f\n" %
                    #     (epoch,
                    #     i + 1,
                    #     perplexity,
                    #     accuracy_AA,
                    #     accuracy_peptide),
                    #     file=log_file_handle,
                    #     end="")
                # validation
                
                sbatt_model.eval()
                spectrum_cnn.eval()
                validaion_begin_time = time.time()
                if deepnovo_config.is_sb:
                    validation_perplexity, accuracy_AA, accuracy_peptide  = validation(spectrum_cnn, valid_data_loader, len(valid_set),
                                                                                        None, None, sbatt_model)
                # else:
                #     validation_perplexity, accuracy_AA, accuracy_peptide  = validation(spectrum_cnn, valid_data_loader, len(valid_set),
                #                                                                         forward_deepnovo, backward_deepnovo, None)
                validation_duration = time.time() - validaion_begin_time
                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train perplexity: {perplexity}\tstep time: {step_time} "
                            f"validation loss: {validation_perplexity}")
                logger.info(f"validation cost: {validation_duration} seconds")
                with open(log_file, 'a') as log_file_handle:
                    print("%.4f \t %.4f \t %.4f\n" % (validation_perplexity, accuracy_AA, accuracy_peptide),
                    file=log_file_handle,
                    end="")
                start_time = time.time()
                if validation_perplexity < 100:
                    if validation_perplexity < best_valid_loss:
                        no_update_count = 0
                        best_valid_loss = validation_perplexity
                        logger.info(f"best valid loss achieved at epoch {epoch} step {i}")
                        best_epoch = epoch
                        best_step = i
                    # save_model(spectrum_cnn, optimizer, None, None, sbatt_model, epoch=epoch)
                    # else:
                    #     save_model(spectrum_cnn, optimizer, forward_deepnovo, backward_deepnovo, None)
                else:
                    no_update_count += 1
                    if no_update_count >= deepnovo_config.early_stop:
                        logger.info(f"early stop at epoch {epoch} step {i}")
                        break

                # back to train model
                spectrum_cnn.train()
                if deepnovo_config.is_sb:
                    sbatt_model.train()
               
        logger.info(f"epoch {epoch} finished")
        save_model(spectrum_cnn, optimizer, None, None, sbatt_model, epoch=epoch)
        # if no_update_count >= deepnovo_config.early_stop:
        #     break

    logger.info(f"best model at epoch {best_epoch} step {best_step}")
