import pickle
import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional, layer
from tqdm import tqdm
import numpy as np




### class definition

## define the integrate but dont fire neuron
class INoFire(neuron.BaseNode):
    '''
    This is the integrate but dont fire neuron, it does not fire but integrate the input
    '''
    def neuronal_charge(self, x: torch.Tensor):
        self.v += x
        return self.v

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = torch.zeros_like(x)
        return spike
class individual_SNN_4_ensemble_no_fire(nn.Module):
    """
    This defines the simple SNN model for the ensemble model, it takes the neuron that does not fire as the output layer
    """

    def __init__(self, input_size, threshold: list, use_bias=False):
        super(individual_SNN_4_ensemble_no_fire, self).__init__()
        self.each_individual = nn.Sequential(
            layer.Linear(input_size, 40, bias=use_bias),
            neuron.IFNode(v_threshold=threshold[0], v_reset=None),
            layer.Linear(40, 18, bias=use_bias)
        )
        self.output_lay = INoFire()

    def forward(self, input_data: torch.Tensor):
        inter_y = self.each_individual(input_data)
        y = self.output_lay(inter_y)
        return y




### function definition

def create_dataset(data, label, batch_size, shuffle=True):
    """
    :param data: the dataset to be loaded
    :param label: the label to be loaded
    :param batch_size: batch size
    :param shuffle: if the dataset should be shuffled
    :return: the dataset holder
    """
    data_tensor = torch.from_numpy(data).float()
    label_tensor = torch.from_numpy(label)
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    data_holder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_holder

def evaluate_accuracy_with_nofire_neuron(data_loader, model, device, T=1, precharge=False):
    """
    This is a new implementation of the evaluation function for the model, which is newly defined SNN model with no fire at the output layer.
    :param data_loader:
    :param model:
    :param device:
    :param T:
    :param precharge:
    :return:
    """
    total = 0
    model.eval()
    loop = tqdm(data_loader)
    corrects = np.zeros(T)
    predicitons = []
    mem_vol = []
    with torch.no_grad():
        for index, (img, label) in enumerate(loop):
            out_v = np.zeros((T, len(img), 18))
            loop.set_description('Evaluating')
            img = img.to(device)
            label = label.to(device)
            if not precharge:
                functional.reset_net(model)
            else:
                for m in model.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                        try:
                            m.v = 0.5 * m.v_threshold
                        except:
                            pass
            for time_step in range(T):
                y = model(img)
                out_v[time_step] = model.output_lay.v.cpu().numpy()
                corrects[time_step] += np.sum(np.argmax(out_v[time_step], axis=1) == label.cpu().numpy()).item()
            mem_vol.append(out_v[-1])
            predicitons.append(np.argmax(out_v[-1], axis=1))
            total += len(label)
    #print(corrects * 100 / total)
    return corrects / total * 100, np.concatenate(predicitons), np.concatenate(mem_vol)

def load_and_eval(index, device, test_dataset, T=10, normed_weights_SNN=None, threshold=None):
    ## directly load the converted SNN model
    if normed_weights_SNN is None:
        with open("/Users/shouyuxie/Projects_machine_learning/ensemble_learning/Ensemble_SNN_HW/Weights_and_data/quant_pruned_SNN/diagonal_" + str(index) + "/quantised_weights.pkl", "rb") as snn_f:
            normed_weights_snn = pickle.load(snn_f)
    if threshold is None:
        with open(
                "/Users/shouyuxie/Projects_machine_learning/ensemble_learning/Ensemble_SNN_HW/Weights_and_data/quant_pruned_SNN/diagonal_" + str(
                        index) + "/quantised_threshold.pkl", "rb") as theta_f:
            threshold = pickle.load(theta_f)

    snn_model = individual_SNN_4_ensemble_no_fire(1023 - index, threshold, use_bias=False)
    state_dict = snn_model.state_dict()
    key_lst = list(state_dict.keys())
    for index in range(len(key_lst)):
        state_dict[key_lst[index]] = torch.from_numpy(normed_weights_snn[index])
    snn_model.load_state_dict(state_dict)
    net = snn_model.to(device)
    acc, predictions, _ = evaluate_accuracy_with_nofire_neuron(test_dataset, net, device, T=T, precharge=True)

    return predictions, acc



def major_hardvote(final_class_of_each_ANN, total_testing_number):
    #final_class_of_each_ANN = np.argmax(all_diagonal_prediction, axis=2)
    final_class_of_each_ANN = np.asarray(final_class_of_each_ANN, 'int')
    final_prediction_after_voting = np.zeros(total_testing_number, 'int')
    for sample_index in range(total_testing_number):
        bin_count = np.bincount(final_class_of_each_ANN[:, sample_index])
        final_prediction_after_voting[sample_index] = np.argmax(bin_count)

    return final_prediction_after_voting