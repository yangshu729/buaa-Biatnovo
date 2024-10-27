import imp
from warnings import catch_warnings
from six.moves import xrange  # pylint: disable=redefined-builtin
import Biatnovo.deepnovo_config as deepnovo_config
import Biatnovo.deepnovo_config_dda as deepnovo_config_dda
import numpy as np
import torch


def get_batch_2(index_list, data_set, bucket_id):
    """TODO(nh2tran): docstring."""

    # ~ print("get_batch()")

    batch_size = len(index_list)
    # batch_size = 32
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
        # 这里返回的decode_input_forward和decoder_input_backward 都是12个数字（此时bucket = 12），数字在0-25之间 Peptide使用数字作为编码
        # decoder_input_forward == peptide_ids_forward 为肽段补齐为12个的编码
        if spectrum_holder is None:  # spectrum_holder is not provided if not use_lstm
            spectrum_holder = np.zeros(shape=(deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE), dtype=np.float32)
        spectrum_holder_list.append(spectrum_holder)
        candidate_intensity_lists_forward.append(candidate_intensity_list_forward)  # --> (batchsize, 12, 26, 40, 10)
        candidate_intensity_lists_backward.append(candidate_intensity_list_backward)
        decoder_inputs_forward.append(decoder_input_forward)  # --> (batchsize, 12)
        decoder_inputs_backward.append(decoder_input_backward)
    # 将其封装成为一个batch
    batch_spectrum_holder = np.array(spectrum_holder_list)
    batch_intensity_inputs_forward = []
    batch_intensity_inputs_backward = []
    batch_decoder_inputs_forward = []
    batch_decoder_inputs_backward = []
    batch_weights = []
    # 此时肽段的统一长度
    decoder_size = deepnovo_config._buckets[bucket_id]
    # decoder_size为12（肽段的统一长度为12，一些大于12的肽段丢弃）
    for length_idx in xrange(decoder_size):
        # batch_intensity_inputs and batch_decoder_inputs are re-indexed.
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
            # The corresponding target is decoder_input shifted by 1 forward.
            # target 为预测的下一个肽段字母，一个batch一起预测，预测的是序列中相同位置的肽段字母
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
    # 变为一个bach的组合，batchsize = 32， 长度为32
    return (
        batch_spectrum_holder,
        batch_intensity_inputs_forward,
        batch_intensity_inputs_backward,
        batch_decoder_inputs_forward,
        batch_decoder_inputs_backward,
        batch_weights,
    )


def get_batch_2_dda(index_list, data_set, bucket_id):
    """TODO(nh2tran): docstring."""

    # ~ print("get_batch()")

    batch_size = len(index_list)
    # batch_size = 32
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
        # 这里返回的decode_input_forward和decoder_input_backward 都是12个数字（此时bucket = 12），数字在0-25之间 Peptide使用数字作为编码
        # decoder_input_forward == peptide_ids_forward 为肽段补齐为12个的编码
        if spectrum_holder is None:  # spectrum_holder is not provided if not use_lstm
            spectrum_holder = np.zeros(
                shape=(deepnovo_config_dda.neighbor_size, deepnovo_config_dda.MZ_SIZE), dtype=np.float32
            )
        spectrum_holder_list.append(spectrum_holder)
        candidate_intensity_lists_forward.append(candidate_intensity_list_forward)  # --> (batchsize, 12, 26, 40, 10)
        candidate_intensity_lists_backward.append(candidate_intensity_list_backward)
        decoder_inputs_forward.append(decoder_input_forward)  # --> (batchsize, 12)
        decoder_inputs_backward.append(decoder_input_backward)
    # 将其封装成为一个batch
    batch_spectrum_holder = np.array(spectrum_holder_list)
    batch_intensity_inputs_forward = []
    batch_intensity_inputs_backward = []
    batch_decoder_inputs_forward = []
    batch_decoder_inputs_backward = []
    batch_weights = []
    # 此时肽段的统一长度
    decoder_size = deepnovo_config_dda._buckets[bucket_id]
    # decoder_size为12（肽段的统一长度为12，一些大于12的肽段丢弃）
    for length_idx in xrange(decoder_size):
        # batch_intensity_inputs and batch_decoder_inputs are re-indexed.
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
            # The corresponding target is decoder_input shifted by 1 forward.
            # target 为预测的下一个肽段字母，一个batch一起预测，预测的是序列中相同位置的肽段字母
            if length_idx < decoder_size - 1:
                target = decoder_inputs_forward[batch_idx][length_idx + 1]
            # We set weight to 0 if the corresponding target is a PAD symbol.
            if (
                length_idx == decoder_size - 1
                or target == deepnovo_config_dda.EOS_ID
                or target == deepnovo_config_dda.GO_ID
                or target == deepnovo_config_dda.PAD_ID
            ):
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    # 变为一个bach的组合，batchsize = 32， 长度为32
    return (
        batch_spectrum_holder,
        batch_intensity_inputs_forward,
        batch_intensity_inputs_backward,
        batch_decoder_inputs_forward,
        batch_decoder_inputs_backward,
        batch_weights,
    )


def test_logit_single_2_score(
    decoder_input_forward, decoder_input_backward, output_logit_forward, output_logit_backward
):
    """TODO(nh2tran): docstring."""

    # length excluding FIRST_LABEL & LAST_LABEL
    # 终止符号的索引
    # print("decoder input forward:", decoder_input_forward) # BOS .. first word... EOS PAD..
    decoder_input_len = decoder_input_forward[1:].index(deepnovo_config.EOS_ID)
    # print("decoder input len:", decoder_input_len)

    output_forward = []
    output_backward = []
    prob_forward = []
    prob_backward = []
    # average forward-backward prediction logit
    # 去掉结束的符号 (seq_len, 26)
    logit_forward = output_logit_forward[:decoder_input_len].cpu().detach().numpy()  # first word ... last word
    # logit_forward = output_logit_forward[:decoder_input_len]  # first word ... last word
    # print("logit_forward:", logit_forward)
    logit_backward = output_logit_backward[:decoder_input_len].cpu().detach().numpy()
    # logit_backward = output_logit_backward[:decoder_input_len]
    # print("output_logit_forward", output_logit_forward)
    # print("output_logit_backward", output_logit_backward)
    logit_backward = logit_backward[::-1]
  
    for x, y in zip(logit_forward, logit_backward):
        prob_forward.append(np.exp(x) / np.sum(np.exp(x)))  # 26个字母中出现的概率
        prob_backward.append(np.exp(y) / np.sum(np.exp(y)))
        # print(prob_forward)
    # print("prob forward:", prob_forward)
    for i in range(decoder_input_len):
        deco = decoder_input_forward[i + 1]
        # print("deco:", deco)
        # print(prob_forward[i])
        output_forward.append(prob_forward[i][deco])
        output_backward.append(prob_backward[i][deco])
    # print("feature index:", feature_index)
    # print("decoder_input_forward", decoder_input_forward[1:decoder_input_len + 1])
    # print("output forward:", output_forward)
    # print("output backward:", output_backward)
    # print("decoder input forward:", decoder_input_forward)
    # print("decoder input backward:", decoder_input_backward)

    import math

    score_forward = sum([math.log(i) for i in output_forward]) / (len(output_forward) + 1e-8)
    # score_forward = sum([i for i in output_forward]) / len(output_forward)
    score_backward = sum([math.log(i) for i in output_backward]) / (len(output_backward) + 1e-8)
    # score_backward = sum([i for i in output_backward]) / len(output_backward)
    score_sum = score_forward + score_backward
    # print("score forward: ", score_forward)
    # print("score backward: ", score_backward)
    # print("score sum: ", score_sum)
    aaid_sequence = decoder_input_forward[1 : decoder_input_len + 1]
    decoder_string = [deepnovo_config.vocab_reverse[x] for x in decoder_input_forward[1 : decoder_input_len + 1]]
    # print("decoder string:", decoder_string)
    position_score = [math.log(i) + math.log(j) for i, j in zip(output_forward, output_backward)]
    # print("score backward:", [math.log(i) for i in output_backward])
    # print("length output forward:", len(output_backward))
    # print("-------------------------------------------------------")
    # output_sum = sum(output)
    # output = [str(x) for x in output]
    # 写入txt中
    # file = '/home2/wusiyu/DeepNovo_data/true_feeding_exchange_order_16.txt'
    # with open(file, 'a+') as f:
    #   line = str(feature_index) + "\t" + "".join(decoder_string) + \
    #          "\t" + str(score_forward) + "\t" + str(score_backward) +\
    #          "\t" + str(score_sum) + "\n"
    #   f.write(line)

    return decoder_string, score_sum, position_score, aaid_sequence
    # return test_AA_true_feeding_single(decoder_input_forward, output, direction=0)


def test_logit_batch_2_score(
    decoder_inputs_forward, decoder_inputs_backward, output_logits_forward, output_logits_backward
):
    output_batch = []
    batch_size = len(decoder_inputs_forward[0])
    index = 0
    # print("output logits forward:", output_logits_forward.shape)
    seq_len_begin = 0
    seq_len = len(output_logits_backward) // batch_size
    assert len(output_logits_backward) % batch_size == 0
    seq_len_end = seq_len
    for batch in xrange(batch_size):
        decoder_input_forward = [x[batch] for x in decoder_inputs_forward]  # (bucket)
        decoder_input_backward = [x[batch] for x in decoder_inputs_backward]  # (bucket)

        # print(output_logits_forward)
        dec_string, score_sum, position_score, aaid_sequence = test_logit_single_2_score(
            decoder_input_forward,
            decoder_input_backward,
            output_logits_forward[seq_len_begin:seq_len_end, :],
            output_logits_backward[seq_len_begin:seq_len_end, :],
        )
        seq_len_begin = seq_len_end
        seq_len_end += seq_len
        output = {}
        output["dec_string"] = dec_string
        output["score_sum"] = score_sum
        output["position_score"] = position_score
        output["aaid_sequence"] = aaid_sequence
        output_batch.append(output)
        index += 1
    # print("output_batch:", output_batch)
    return output_batch


def test_accuracy_score(model, data_set, bucket_id, opt):
    data_set_len = len(data_set[bucket_id])
    data_set_index_list = range(data_set_len)
    data_set_index_chunk_list = [
        data_set_index_list[i : i + deepnovo_config.batch_size_predict]
        for i in range(0, data_set_len, deepnovo_config.batch_size_predict)
    ]
    index = 0
    bi_score = []
    for chunk in data_set_index_chunk_list:
        (
            spectrum_holder,
            intensity_inputs_forward,
            intensity_inputs_backward,
            decoder_inputs_forward,
            decoder_inputs_backward,
            target_weights,
        ) = get_batch_2(chunk, data_set, bucket_id)

        decoder_size = deepnovo_config._buckets[bucket_id]
        # print("this max candidate intensity:", torch.max(torch.tensor(intensity_inputs_forward[0][0]), dim=1))

        # print("spectrum_holder:", spectrum_holder[0][2][8000:8500])

        spectrum_holder = torch.from_numpy(spectrum_holder).cuda()
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
        # output_logits_forward( seq len, 26)
        # decoder_inputs_forward( seq len, batch size)
        output_batch = test_logit_batch_2_score(
            decoder_inputs_forward[:decoder_size],
            decoder_inputs_backward[:decoder_size],
            output_logits_forward,
            output_logits_backward,
        )

        index += 1
        for i in range(len(output_batch)):
            tmp = output_batch[i]
            bi_score.append(tmp)
        # print("bi score:", bi_score)

    return bi_score


def test_accuracy_score_dda(model, data_set, bucket_id, opt):
    data_set_len = len(data_set[bucket_id])
    # feature_index = bucket_feature[bucket_id]
    data_set_index_list = range(data_set_len)
    data_set_index_chunk_list = [
        data_set_index_list[i : i + deepnovo_config_dda.batch_size_predict]
        for i in range(0, data_set_len, deepnovo_config_dda.batch_size_predict)
    ]
    index = 0
    bi_score = []
    for chunk in data_set_index_chunk_list:
        (
            spectrum_holder,
            intensity_inputs_forward,
            intensity_inputs_backward,
            decoder_inputs_forward,
            decoder_inputs_backward,
            target_weights,
        ) = get_batch_2_dda(chunk, data_set, bucket_id)

        decoder_size = deepnovo_config_dda._buckets[bucket_id]
        # print("this max candidate intensity:", torch.max(torch.tensor(intensity_inputs_forward[0][0]), dim=1))

        # print("spectrum_holder:", spectrum_holder[0][2][8000:8500])

        spectrum_holder = torch.from_numpy(spectrum_holder).cuda()
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
        # output_logits_forward( seq len, 26)
        output_batch = test_logit_batch_2_score(
            decoder_inputs_forward[:decoder_size],
            decoder_inputs_backward[:decoder_size],
            output_logits_forward,
            output_logits_backward,
        )

        index += 1
        for i in range(len(output_batch)):
            tmp = output_batch[i]
            bi_score.append(tmp)
        # print("bi score:", bi_score)
    # print(bi_score)
    return bi_score
