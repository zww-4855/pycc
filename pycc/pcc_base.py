import numpy as np
import pycc.tamps as tamps

def build_MP2_T2(W,D2):
    roovv = 0.250000000 * np.einsum("ijab->ijab",W,optimize="optimal")
    roovv = tamps.antisym_T2(roovv,None,None)
    roovv = roovv*D2
    return roovv



def build_LCCD_T2(T2,W,o,v,D2):
    roovv = 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")

    roovv = tamps.antisym_T2(roovv,None,None)
    roovv = roovv*D2
    return roovv


def get_WnT2_energy(T2,g):
    energy = 0.250000000 * np.einsum("ijab,abij->",T2,g,optimize="optimal")
    return energy

def kill_Diag_T2(roovv,nocc,nvirt):
    for i in range(nocc-1):
        for a in range(nvirt-1):
            roovv[i,i+1,a,a+1]=0.0
            roovv[i+1,i,a,a+1]=0.0
            roovv[i,i+1,a+1,a]=0.0
            roovv[i+1,i,a+1,a]=0.0

    return roovv

def return_Diag_T2(roovv,nocc,nvirt):
    tmpT2=np.zeros((nocc,nocc,nvirt,nvirt))
    for i in range(nocc-1):
        for a in range(nvirt-1):
            tmpT2[i,i+1,a,a+1] = roovv[i,i+1,a,a+1]
            tmpT2[i+1,i,a,a+1] = roovv[i+1,i,a,a+1]
            tmpT2[i,i+1,a+1,a] = roovv[i,i+1,a+1,a]
            tmpT2[i+1,i,a+1,a] = roovv[i+1,i,a+1,a]

    return tmpT2

def return_Diag_Identity(roovv,nocc,nvirt):
    tmpT2=np.zeros((nocc,nocc,nvirt,nvirt))
    for i in range(nocc-1):
        for a in range(nvirt-1):
            tmpT2[i,i+1,a,a+1] = 1.0# roovv[i,i+1,a,a+1]
            tmpT2[i+1,i,a,a+1] = -1.0 #roovv[i+1,i,a,a+1]
            tmpT2[i,i+1,a+1,a] = 1.0 #roovv[i,i+1,a+1,a]
            tmpT2[i+1,i,a+1,a] = -1.0#roovv[i+1,i,a+1,a]

    return tmpT2

