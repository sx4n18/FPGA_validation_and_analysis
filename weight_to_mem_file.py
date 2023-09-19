import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split

############################################################################################
## This script will try to take the weights and write it in binary format in a mem file
## The mem file will be used to load the weights in the verilog code
## "input_value_int.mem"
## "offset_mem.mem"
## "CSR_weight.mem"
## "weight_index.mem"
## "CSC_weight.mem"
############################################################################################

def number2_complement(number, bits):
    if number < 0:
        complement_rep = (1 << bits) + number
        return bin(complement_rep)[2:]
    else:
        complement_rep = number
        return bin(complement_rep)[2:].zfill(bits)

net_index = 3
## load the sparsed weight matrix
with open("./quant_pruned_SNN/diagonal_"+str(net_index)+"/quantised_weights.pkl", "rb") as quant_w_f:
    quant_w = pickle.load(quant_w_f)

first_layer_w = np.array(quant_w[0], dtype=np.int32)
second_layer_w = np.array(quant_w[1], dtype=np.int32)
list_of_weights = []
list_of_index = []
list_of_offset = []

list_of_second_weights = []
list_of_index_second = [0]
list_of_col_index = []
neuron_id_cnt = 0

for i in range(40):
    first_row_w = first_layer_w[i]
    second_col_w = second_layer_w[:, i]
    list_of_offset.append(np.count_nonzero(first_row_w))
    neuron_id_cnt += np.count_nonzero(second_col_w)
    list_of_index_second.append(neuron_id_cnt)

    list_of_col_index.append(np.nonzero(second_col_w)[0])
    list_of_index.append(np.nonzero(first_row_w)[0])
    list_of_weights.append(first_row_w[np.nonzero(first_row_w)])
    list_of_second_weights.append(second_col_w[np.nonzero(second_col_w)])

list_of_weights = list(np.concatenate(list_of_weights))
list_of_index = list(np.concatenate(list_of_index))

list_of_col_index = list(np.concatenate(list_of_col_index))
list_of_second_weights = list(np.concatenate(list_of_second_weights))

## load the input value matrix
data_sample_int = np.load("./log_2_plus_1_subtraction_prepro_int/log_2_data_"+str(net_index)+"_int.npy")
label = np.load("data_set/label.npy")
label = label -1
Train_X, Test_X, Train_y, Test_y = train_test_split(data_sample_int, label, test_size=0.15, random_state=42,
                                                                      shuffle=True)

First_sample = np.array(Test_X[18], dtype=np.int32)
first_sample_label = Test_y[0]
print("First sample label: ", first_sample_label)
sum = np.matmul(First_sample, first_layer_w.T)

sample_non_zero = np.nonzero(First_sample)[0]
first_layer_non_zero = np.nonzero(first_layer_w[0])[0]
actual_computation = []
for index in range(1023):
    if index in sample_non_zero and index in first_layer_non_zero:
        actual_computation.append(index)

print(actual_computation)
#print(first_layer_w[0][60])
#print(First_sample[60])
print("\nTime step 0:")
print(sum+63)
temp_sum = sum+63
print("Spike in time step 0:")
print(np.where(temp_sum>127))
spike_AER = np.where(temp_sum>127, 1, 0)
temp_sum = np.where(temp_sum >127, temp_sum-127, temp_sum)
print("Activation after time step 0:")
print(temp_sum)
second_activation = np.matmul(spike_AER, second_layer_w.T)
print("Second layer activation:")
print(second_activation)

print("\nTime step 1:")
temp_sum += sum
print("Spike in time step 1:")
print(np.where(temp_sum>127))
spike_AER = np.where(temp_sum>127, 1, 0)
temp_sum = np.where(temp_sum >127, temp_sum-127, temp_sum)
print("Activation after time step 0:")
print(temp_sum)
second_activation += np.matmul(spike_AER, second_layer_w.T)
print("Second layer activation:")
print(second_activation)

print("\nTime step 2:")
temp_sum += sum
print("Spike in time step 2:")
print(np.where(temp_sum>127))
spike_AER = np.where(temp_sum>127, 1, 0)
temp_sum = np.where(temp_sum >127, temp_sum-127, temp_sum)
print("Activation after time step 0:")
print(temp_sum)
second_activation += np.matmul(spike_AER, second_layer_w.T)
print("Second layer activation:")
print(second_activation)


print("\nTime step 3:")
temp_sum += sum
print("Spike in time step 3:")
print(np.where(temp_sum>127))
spike_AER = np.where(temp_sum>127, 1, 0)
temp_sum = np.where(temp_sum >127, temp_sum-127, temp_sum)
print("Activation after time step 0:")
print(temp_sum)
second_activation += np.matmul(spike_AER, second_layer_w.T)
print("Second layer activation:")
print(second_activation)

print("Winner neuron:")
print(np.argmax(second_activation))

#if not os.path.exists("../mem"):
#    os.makedirs("../mem")

#with open("../mem/offset_mem.mem", "w") as offset_f:
#    for offset in list_of_offset:
#        offset_f.writelines(bin(offset)[2:].zfill(10)+'\n')

#with open("../mem/CSR_weight.mem", "w") as csr_w_f:
#    for index, weight in zip(list_of_index, list_of_weights):
#        csr_w_f.writelines(number2_complement(index, 10) + number2_complement(weight, 8)+'\n')


#with open("../mem/input_value_int.mem", "w") as input_f:
#    for input in First_sample:
#        input_f.writelines(number2_complement(input, 8) + "\n")

#with open("../mem/weight_index.mem", "w") as weight_index_f:
#    for index in list_of_index_second:
#        weight_index_f.writelines(number2_complement(index, 8) + "\n")

#with open("../mem/CSC_weight.mem", "w") as csc_weight_f:
#    for col_index, weight in zip(list_of_col_index, list_of_second_weights):
#            csc_weight_f.writelines(number2_complement(col_index, 5) + number2_complement(weight, 8) + "\n")
