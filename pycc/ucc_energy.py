import numpy as np
import pycc.ucc_eqns as ucc_eqns
import pycc.tamps as tamps
import pycc.faster_ucc5eqns as faster_ucc5eqns

def ucc_energyDriver(calcType,W,T1,T2,o,v,driveCCobj):
    energy = 0.0
    if calcType == "UCCD3":
        return 0.250000000 * np.einsum("ijab,abij->",T2,W[v,v,o,o],optimize="optimal")

    if "UCCSD4" in calcType or "UCCD4" in calcType:
        D2 = driveCCobj.denomInfo["D2aa"]
        energy = uccsd4_energy(W,T2,o,v,D2)


    if "UCCSD5" in calcType or "UCCD5" in calcType:
        D2 = driveCCobj.denomInfo["D2aa"]
        energy = uccsd4_energy(W,T2,o,v,D2)
        energy += uccsd5_energy(W,T1,T2,o,v)
        # Get remaining UCCSD5/UCCD5 terms

    return energy



def uccsd4_energy(W,T2,o,v,D2):
    # <0|WnT2|0>
    energyA = 0.250000000 * np.einsum("ijab,abij->",T2,W[v,v,o,o],optimize="optimal")

    D2T2 = ucc_eqns.uccsd_T2dagWnT2(W,T2,o,v)
    nocc=nvir=None
    D2T2=tamps.antisym_T2(D2T2,nocc,nvir)
    T2dag = T2.transpose(2,3,0,1) 
    energyB = 0.250000000 * np.einsum("ijab,abij->",D2T2,T2dag,optimize="optimal")
    print('energy A/B:',energyA,energyB)


    # Now build 0.25*<0|(T2^)^2WnT2|0>, as in Watts papers

    # 0/5* [T2^,[T2^,Wn]],T2]
    r = 1.000000000 * np.einsum("ijab,acik,bdjl,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.500000000 * np.einsum("ijab,cdjk,abil,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.125000000 * np.einsum("ijab,abkl,cdij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.500000000 * np.einsum("ijab,bckl,adij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    oldr=r/4
    print('r vs B:',r, energyB)

    # (1/3!) * [[[H,tau2],tau2],tau2]
    r = 0
    r = 0.333333333 * np.einsum("ijab,acik,bdjl,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.166666667 * np.einsum("ijab,cdjk,abil,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.041666667 * np.einsum("ijab,abkl,cdij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.166666667 * np.einsum("ijab,bckl,adij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    print('newr/oldr', r,(6.0/4.0)*r,oldr,(3.0/4.0)*r)
    return energyA-oldr #+energyB

def uccsd5_energy(W,T1,T2,o,v):
    T2dag = T2.transpose(2,3,0,1)
    # - <0|(T1^T2^W)CT2|0>
    D2T2 = ucc_eqns.uccsd5_t1dagwnCt2(W,T1,T2,o,v)
    D2T2 = tamps.antisym_T2(D2T2,None,None)
    ucc5_energy = - 0.250000000 * np.einsum("ijab,abij->",D2T2,T2dag,optimize="optimal")

    # - 0.25* <0|(T2^)^2 (WT2^2)C|0>
    #D2T2 = ucc_eqns.uccsd5_t2resid_t2dag_wnt2sqrC(W,T2,o,v)
    D2T2 = faster_ucc5eqns.t2dag_WnT2sqr_oooo(T2,W[o,o,o,o])
    D2T2 += faster_ucc5eqns.t2dag_WnT2sqr_ovov(T2,W[o,v,o,v])
    D2T2 += faster_ucc5eqns.t2dag_WnT2sqr_vvvv(T2,W[v,v,v,v])

    D2T2 = tamps.antisym_T2(D2T2,None,None)
    ucc5_energy += - (0.25)* 0.250000000 * np.einsum("ijab,abij->",D2T2,T2dag,optimize="optimal")
    return ucc5_energy

