from func.SJ_model_fun import load_and_eval, create_dataset, major_hardvote
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

############################################################################################
## This script will rerun the spiking jelly simulation with the weights and threshold loaded
## in FPGA and compare the results with the FPGA results
############################################################################################

## load the results from the FPGA and ground truth
fpga_result = np.load("./FPGA_test_after_log_fix/ans_record.npy")
ground_truth = np.load("./FPGA_test/Ground_truth_label.npy")

## parameters definition
test_sample_num = int(9000*0.15)
number_of_net = 5
T = 4
total_cls_num = 18
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
label = np.load("./data_set/label.npy")-1
each_prediciton = np.zeros((number_of_net, test_sample_num))

## loop through all the networks
for net_index in range(number_of_net):
    ## load the weights and threshold from pickle files
    #with open("./quant_pruned_SNN/diagonal_"+str(net_index)+"/quantised_weights.pkl", "rb") as snn_f:
    #    normed_weights_snn = pickle.load(snn_f)
    #with open("./quant_pruned_SNN/diagonal_"+str(net_index)+"/quantised_threshold.pkl", "rb") as theta_f:
    #    threshold = pickle.load(theta_f)

    ## load the input value matrix
    data_sample_int = np.load("./log_2_plus_1_subtraction_prepro_int/log_2_data_"+str(net_index)+"_int.npy")

    ## train test split and create dataloader
    train_X, test_X, train_y, test_y = train_test_split(data_sample_int, label, test_size=0.15, random_state=42)
    test_dat_holder = create_dataset(test_X, test_y, batch_size=500, shuffle=False)

    ## Evaluation
    prediction, acc = load_and_eval(net_index, device, test_dat_holder, T)
    each_prediciton[net_index] = prediction


## hard voting
final_prediction = major_hardvote(each_prediciton, test_sample_num)
final_acc = accuracy_score(ground_truth, final_prediction)

## print the results
print("Final accuracy: ", final_acc)

## hardware software inconsisiteny analysis
inconsistency_cnt_FPGA_wrong = 0
inconsistency_cnt_SJ_wrong = 0
correct_cnt = 0
for index in range(test_sample_num):
    if fpga_result[index] != final_prediction[index] and fpga_result[index] != ground_truth[index]:
        inconsistency_cnt_FPGA_wrong += 1
        print("------------------------------------------------------")
        print("Index: ", index)
        print("FPGA result: ", fpga_result[index])
        print("Ground truth: ", ground_truth[index])
        print("SNN software result: ", final_prediction[index])
        print("SNN software each result: ", each_prediciton[:, index])
        print("------------------------------------------------------")
    elif fpga_result[index] != final_prediction[index] and fpga_result[index] == ground_truth[index]:
        inconsistency_cnt_SJ_wrong += 1
        print("------------------------------------------------------")
        print("Index: ", index)
        print("FPGA result: ", fpga_result[index])
        print("SNN software result: ", final_prediction[index])
        print("SNN software each result: ", each_prediciton[:, index])
        print("------------------------------------------------------")
    else:
        correct_cnt += 1


