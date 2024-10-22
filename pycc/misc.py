import numpy as np
from copy import deepcopy

def zeroT2_offDiagonal(tensor):
    o = np.shape(tensor)[0]
    v = np.shape(tensor)[2]
    diagT2 = np.zeros((o,o,v,v))
    for occ in range(o):
        for virt in range(v):
            diagT2[occ,occ,virt,virt]=tensor[occ,occ,virt,virt]
    return diagT2

def zeroT2_Diagonal(tensor):
    o = np.shape(tensor)[0]
    v = np.shape(tensor)[2]
    diagT2 = np.zeros((o,o,v,v))
    diagT2 = deepcopy(tensor)
    for occ in range(o):
        for virt in range(v):
            diagT2[occ,occ,virt,virt]= 0.0 #tensor[occ,occ,virt,virt]
    return diagT2

