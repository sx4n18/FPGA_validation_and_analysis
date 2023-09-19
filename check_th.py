import pickle

### check if all the thresholds are the same


for i in range(20):
    path = "./quant_pruned_SNN/diagonal_" + str(i) + "/quantised_threshold.pkl"
    f = open(path, "rb")
    quant_th = pickle.load(f)
    f.close()
    print(quant_th)

### check weights shape
path = "./quant_pruned_SNN/diagonal_0/quantised_weights.pkl"
f = open(path, "rb")
quant_w = pickle.load(f)
f.close()
print("Total length of weights list: ")
print(len(quant_w))
print("First layer weights shape: ")
print(quant_w[0].shape)
print("Second layer weights shape: ")
print(quant_w[1].shape)