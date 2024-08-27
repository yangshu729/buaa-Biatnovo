import numpy as np
import torch
import torch.nn.functional as F
from v2 import deepnovo_config


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

def cal_dia_focal_loss(pred_forward, pred_backward, gold_forward, gold_backward, batch_size):
    # pred_forward.shape = (batchsize * (decoder_size - 1), num_classes)
    zeros_forward = torch.zeros_like(gold_forward, dtype=gold_forward.dtype)
    ones_forward = torch.ones_like(gold_forward, dtype=gold_forward.dtype)
    # [batch_size, seq_len] ,如果 gold_forward 在某位置的值为 0（假设 0 为填充索引），则 gold_forward_weight 在该位置的值将为 0
    gold_forward_weight = torch.where(gold_forward == 0, zeros_forward, ones_forward)
    zeros_backward = torch.zeros_like(gold_backward, dtype=gold_backward.dtype)
    ones_backward = torch.ones_like(gold_backward, dtype=gold_backward.dtype)
    gold_backward_weight = torch.where(gold_backward == 0, zeros_backward, ones_backward)
    # focal_loss和weight逐元素相乘，loss_f 的 shape 为 (batchsize * (decoder_size - 1),)
    loss_f = cal_folcal_loss(pred_forward, gold_forward) * gold_forward_weight
    total_forward_weight = torch.sum(gold_forward_weight) + 1e-12
    loss_f = torch.sum(loss_f) / total_forward_weight
    loss_b = cal_folcal_loss(pred_backward, gold_backward) * gold_backward_weight
    total_backward_weight = torch.sum(gold_backward_weight) + 1e-12
    loss_b = torch.sum(loss_b) / total_backward_weight
    loss = (loss_b + loss_f) / 2.0
    return loss

def cal_sb_dia_focal_loss(pred_forward, pred_backward, gold_forward, gold_backward, forward_half_mask, backward_half_mask):
    zeros_forward = torch.zeros_like(gold_forward, dtype=gold_forward.dtype)
    ones_forward = torch.ones_like(gold_forward, dtype=gold_forward.dtype)
    gold_forward_weight = torch.where(gold_forward == 0, zeros_forward, ones_forward) * forward_half_mask
    zeros_backward = torch.zeros_like(gold_backward, dtype=gold_backward.dtype)
    ones_backward = torch.ones_like(gold_backward, dtype=gold_backward.dtype)
    gold_backward_weight = torch.where(gold_backward == 0, zeros_backward, ones_backward) * backward_half_mask
    loss_f = cal_folcal_loss(pred_forward, gold_forward) * gold_forward_weight
    total_forward_weight = torch.sum(gold_forward_weight) + 1e-12
    loss_f = torch.sum(loss_f) / total_forward_weight
    loss_b = cal_folcal_loss(pred_backward, gold_backward) * gold_backward_weight
    total_backward_weight = torch.sum(gold_backward_weight) + 1e-12
    loss_b = torch.sum(loss_b) / total_backward_weight
    loss = (loss_b + loss_f) / 2.0
    return loss


def trim_decoder_input(decoder_input, direction):
  """TODO(nh2tran): docstring."""

  if direction == 0:
    LAST_LABEL = deepnovo_config.EOS_ID
  elif direction == 1:
    LAST_LABEL = deepnovo_config.GO_ID

  # excluding FIRST_LABEL, LAST_LABEL & PAD
  return decoder_input[1:decoder_input.index(LAST_LABEL)]

def test_AA_true_feeding_single(decoder_input, output, direction):
  """TODO(nh2tran): docstring."""

  accuracy_AA = 0.0
  len_AA = 0.0
  exact_match = 0.0
  len_match = 0.0

  # decoder_input = [AA]; output = [AA...]
  decoder_input = trim_decoder_input(decoder_input, direction)
  decoder_input_len = len(decoder_input)

  # measure accuracy
  num_match = test_AA_match_1by1(decoder_input, output)
  # ~ accuracy_AA = num_match / decoder_input_len
  accuracy_AA = num_match
  len_AA = decoder_input_len
  if num_match == decoder_input_len:
    exact_match = 1.0
  # ~ if output_len == decoder_input_len:
  # ~ len_match = 1.0

  return accuracy_AA, len_AA, exact_match, len_match

def test_logit_single_2(decoder_input_forward,
                        decoder_input_backward,
                        output_logit_forward,
                        output_logit_backward):
    """TODO(nh2tran): docstring."""

    # length excluding FIRST_LABEL & LAST_LABEL

    decoder_input_len = decoder_input_forward[1:].index(deepnovo_config.EOS_ID)

    # 去掉结束的符号
    logit_forward = output_logit_forward[:decoder_input_len].cpu().detach().numpy()  # first word ... last word
    logit_backward = output_logit_backward[:decoder_input_len].cpu().detach().numpy()
    # logit_backward = output_logit_backward[:decoder_input_len]
    logit_backward = logit_backward[::-1]

    output = []
    for x, y in zip(logit_forward, logit_backward):
      prob_forward = np.exp(x) / np.sum(np.exp(x))
      prob_backward = np.exp(y) / np.sum(np.exp(y))
      output.append(np.argmax(prob_forward * prob_backward))

    output.append(deepnovo_config.EOS_ID)

    return test_AA_true_feeding_single(decoder_input_forward, output, direction=0)

def test_logit_batch_2(decoder_inputs_forward,
                       decoder_inputs_backward,
                       output_logits_forward, # # (seq_len,batchsize, 26))
                       output_logits_backward):
    """TODO(nh2tran): docstring.

    """
    batch_accuracy_AA = 0.0
    batch_len_AA = 0.0
    num_exact_match = 0.0
    num_len_match = 0.0
    batch_size = len(decoder_inputs_forward[0])
    output_logits_forward = output_logits_forward.view(-1, batch_size, deepnovo_config.vocab_size)
    output_logit_backward = output_logits_backward.view(-1, batch_size, deepnovo_config.vocab_size)
    # Process each sample in the batch
    for batch in range(batch_size):
        decoder_input_forward = [x[batch] for x in decoder_inputs_forward]
        decoder_input_backward = [x[batch] for x in decoder_inputs_backward]
        output_logit_forward = output_logits_forward[:, batch, :]  # (seq_len, 26)
        output_logit_backward = output_logits_backward[:, batch, :]

        accuracy_AA, len_AA, exact_match, len_match = test_logit_single_2(
            decoder_input_forward,
            decoder_input_backward,
            output_logit_forward,
            output_logit_backward)

        batch_accuracy_AA += accuracy_AA
        batch_len_AA += len_AA
        num_exact_match += exact_match
        num_len_match += len_match

    return batch_accuracy_AA, batch_len_AA, num_exact_match, num_len_match

def test_AA_match_1by1(decoder_input, output):
  """TODO(nh2tran): docstring."""

  decoder_input_len = len(decoder_input)

  num_match = 0
  index_aa = 0
  while index_aa < decoder_input_len:
    # ~ if  decoder_input[index_aa]==output[index_aa]:
    if (abs(deepnovo_config.mass_ID[decoder_input[index_aa]]
            - deepnovo_config.mass_ID[output[index_aa]])
        < deepnovo_config.AA_MATCH_PRECISION):
      num_match += 1
    index_aa += 1

  return num_match

