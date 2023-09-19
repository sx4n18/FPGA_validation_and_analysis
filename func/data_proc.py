import numpy as np


def number2_complement(number, bits):
    if number < 0:
        complement_rep = (1 << bits) + number
        return bin(complement_rep)[2:]
    else:
        complement_rep = number
        return bin(complement_rep)[2:].zfill(bits)


def give_offset_CSR_mem_file(raw_ndarr):
    list_of_weights = []
    lisf_of_row_idx = []
    list_of_offset = []
    for i in range(40): ## it is always 40 for whichever network
        weight = raw_ndarr[i]
        list_of_offset.append(np.count_nonzero(weight))
        list_of_weights.append(weight[np.nonzero(weight)]) ## get the weihts that are non zero
        lisf_of_row_idx.append(np.nonzero(weight)[0]) ## get the index of the non zero weights
    list_of_weights = list(np.concatenate(list_of_weights))
    lisf_of_idx = list(np.concatenate(lisf_of_row_idx))
    return list_of_weights, lisf_of_idx, list_of_offset

def give_col_idx_CSC_mem_file(raw_ndarr):
    list_of_col_index = []
    list_of_second_weights = []
    list_of_mem_offset = [0]
    neuron_id_cnt = 0
    for i in range(40): ## it is always 40 for whichever network
        col_w = raw_ndarr[:, i]
        neuron_id_cnt += np.count_nonzero(col_w)
        list_of_mem_offset.append(neuron_id_cnt)
        list_of_col_index.append(np.nonzero(col_w)[0])
        list_of_second_weights.append(col_w[np.nonzero(col_w)])
    list_of_col_index = list(np.concatenate(list_of_col_index))
    list_of_second_weights = list(np.concatenate(list_of_second_weights))
    return list_of_second_weights, list_of_col_index, list_of_mem_offset

