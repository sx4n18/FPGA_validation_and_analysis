import numpy as np



############################################################################################
## This script will try to compare the results from the FPGA and the software simulation to
## see where the difference comes from
############################################################################################

## load the results from the FPGA and ground truth
fpga_result = np.load("./FPGA_test_after_log_fix/ans_record.npy")
ground_truth = np.load("./FPGA_test/Ground_truth_label.npy")

## load the results from spiking jelly simulation
each_prediciton = np.load("./SNN_software_results/SNN_ensemble_prediction.npy")
final_prediction = np.load("./SNN_software_results/hard_voting_prediction.npy")

test_sample_num = int(9000*0.15)
inconsistency_cnt = 0

## hardware software inconsisiteny analysis
for index in range(test_sample_num):
    if fpga_result[index] != final_prediction[index] and fpga_result[index] != ground_truth[index]:
        inconsistency_cnt += 1
        print("------------------------------------------------------")
        print("Index: ", index)
        print("FPGA result: ", fpga_result[index])
        print("Ground truth: ", ground_truth[index])
        print("SNN software result: ", final_prediction[index])
        print("SNN software each result: ", each_prediciton[:, index])
        print("------------------------------------------------------")


print("Total number of inconsistency: ", inconsistency_cnt)