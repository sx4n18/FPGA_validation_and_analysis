import numpy as np


def manual_rint_log2(x):
    ceiling = len(bin(x+1))-2
    if (x+1) / (2**(ceiling-1)) >= np.sqrt(2):
        return ceiling
    else:
        return ceiling - 1


for i in range(2**20):
    if np.rint(np.log2(i+1)) != manual_rint_log2(i):
        print("Error")
        print(i)
        print(np.rint(np.log2(i+1)))
        print(manual_rint_log2(i))
        break