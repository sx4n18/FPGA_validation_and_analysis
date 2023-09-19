import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio



############################################################################################
## This script will generate the testbench for the SNN inference with the index given
############################################################################################


## load the data matrix from the matlab file
mat_file = sio.loadmat("./data_set/data.mat")
data_npy = mat_file["data"]
label_npy = np.load("./data_set/label.npy") -1

## split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data_npy, label_npy, test_size=0.15, random_state=42)
test_sample_index = 18
sample = X_test[test_sample_index]
log_2_sample = np.log2(sample + 1)
log_2_sample_int = np.rint(log_2_sample)
## generate the testbench
tb_file = open("./FPGA_test/testbench.txt", "w")

## write the input data
for index in range(1024):
    if index == 0:
        tb_file.write("trans_start = 1;\n")
        tb_file.write("bin_cnt = " + str(sample[index]) + ";\n")
        tb_file.write("#10\n")
    elif index == 1:
        tb_file.write("trans_start = 0;\n")
        tb_file.write("bin_cnt = " + str(sample[index]) + ";\n")
        tb_file.write("#10\n")
    else:
        tb_file.write("bin_cnt = " + str(sample[index]) + ";\n")
        tb_file.write("#10\n")

tb_file.close()