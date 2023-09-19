import pickle
import numpy as np
from .data_proc import number2_complement


## "offset_mem.mem"
## "CSR_weight.mem"
## "weight_index.mem"
## "CSC_weight.mem"

def load_sparse_weight(index_diagonal):
    path = "./quant_pruned_SNN/diagonal_" + str(index_diagonal) + "/quantised_weights.pkl"
    f = open(path, "rb")
    quant_w = pickle.load(f)
    f.close()
    first_layer_w = np.array(quant_w[0], dtype=np.int32)
    second_layer_w = np.array(quant_w[1], dtype=np.int32)
    return first_layer_w, second_layer_w

def write_offset_CSR_mem_file(diagonal, list_of_offset, list_of_weights, list_of_row_index):
    ## convert A to ascii and increment by diagonal
    increment_char = ord('A') + diagonal
    suffix = chr(increment_char)
    path_of_offset = "../mem/offset_mem" + suffix + ".mem"
    path_of_weights = "../mem/CSR_weight" + suffix + ".mem"
    with open(path_of_offset, "w") as f_o:
        for offset in list_of_offset:
            f_o.write(bin(offset)[2:].zfill(10)+'\n')

    with open(path_of_weights, "w") as f_w:
        for index, weight in zip(list_of_row_index, list_of_weights):
            f_w.writelines(number2_complement(index, 10) + number2_complement(weight, 8) + '\n')

def write_col_idx_CSC_mem_file(diagonal, list_of_col_index, list_of_second_weights, list_of_mem_offset):
    ## convert A to ascii and increment by diagonal
    increment_char = ord('A') + diagonal
    suffix = chr(increment_char)
    path_of_weight_idx = "../mem/weight_index" + suffix + ".mem"
    path_of_second_weights = "../mem/CSC_weight" + suffix + ".mem"
    with open(path_of_weight_idx, "w") as f_wid:
        for w_index in list_of_mem_offset:
            f_wid.write(number2_complement(w_index, 8) + "\n")

    with open(path_of_second_weights, "w") as f_sw:
        for index, weight in zip(list_of_col_index, list_of_second_weights):
            f_sw.write(number2_complement(index, 5) + number2_complement(weight, 8) + '\n')
