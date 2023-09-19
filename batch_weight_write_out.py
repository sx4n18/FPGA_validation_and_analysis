from func import data_proc
from func import File_acc

##############################################################################################################
## This script will batch write out the mem files for each diagonal
## The mem file will be used to load the weights in the verilog code
## steps as follow:
## 1. load the sparsed weight matrix to two ndarrays where each one stands for one layer
## 2. extract all the essential information in the sparse matrix. i.e. CSR and CSC compression
## 3. write out the mem files
##############################################################################################################



for diagonal in range(1,20):
    ## load the sparsed weight matrix
    first_layer_w, second_layer_w = File_acc.load_sparse_weight(diagonal)

    ## extract all the essential information in the sparse matrix. i.e. CSR and CSC compression
    list_of_weights, list_of_index, list_of_offset = data_proc.give_offset_CSR_mem_file(first_layer_w)
    list_of_second_weights, list_of_col_index, list_of_mem_offset = data_proc.give_col_idx_CSC_mem_file(second_layer_w)

    ## write out the mem files
    File_acc.write_offset_CSR_mem_file(diagonal, list_of_offset, list_of_weights, list_of_index)
    File_acc.write_col_idx_CSC_mem_file(diagonal, list_of_col_index, list_of_second_weights, list_of_mem_offset)

    print("Diagonal " + str(diagonal) + " mem files are written out")