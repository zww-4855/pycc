import pycc
import numpy as np
import pycc.wicked_T3corr

def build_penalty_op(F,W,T2,o,v,D2,D3):
    T3SO = build_Fn_WT2_to_T3(W,T2,o,v,D3)
    T3SO += build_SO_Fn_T3(F,T3SO,o,v,D3)

    # Now, contract Q2(WT3)
    D2T2eff = build_effectiveT2_from_T3(W,T3SO,D2,o,v)
    # verify I get [T] 
    T2eff = D2T2eff*D2
    #penalty_op_T2eff = pycc.pcc_base.get_WnT2_energy(T2eff,T2eff.transpose(2,3,0,1))  
    print('shape of t2eff',T2eff.shape,F[o,o].shape,T3SO.shape)
    return T2eff



def build_Fn_WT2_to_T3(W,T2,o,v,D3):
    D3T3 = pycc.build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,T2)
    D3T3 = pycc.tamps.antisym_T3(D3T3,None,None)
    T3 = D3T3*D3 
    return T3

def build_SO_Fn_T3(F,T3,o,v,D3):
    D3_FnT3 = -0.083333333 * np.einsum("il,jklabc->ijkabc",F[o,o],T3,optimize="optimal")
    D3_FnT3 += 0.083333333 * np.einsum("da,ijkbcd->ijkabc",F[v,v],T3,optimize="optimal")
    D3T3 = pycc.tamps.antisym_T3(D3_FnT3,None,None)
    T3 = D3T3*D3
    return T3

def build_effectiveT2_from_T3(W,T3,D2,o,v):
    netD2T2 = pycc.wicked_T3corr.build_netT2(W,o,v,T3)
    netD2T2 = pycc.tamps.antisym_T2(netD2T2,None,None)*D2
    return netD2T2


def uccsd_FO_triples_corrections(F,W,T2,o,v,D3):
    D3T3 = pycc.build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,T2)
    D3T3 = pycc.tamps.antisym_T3(D3T3,None,None)
    T3 = D3T3*D3 #.transpose(3,4,5,0,1,2)
    sqrBrak_T =0.25* pycc.build_sqrbrak_corrections.sqr_brakT_spin(D3T3,T3.transpose(3,4,5,0,1,2))

    total_FO_trples = sqrBrak_T +  pycc.build_sqrbrak_corrections.build_T3dag_fn_T3(T3,F,o,v)
    print("2*[T] vs T3fnT3:",sqrBrak_T,pycc.build_sqrbrak_corrections.build_T3dag_fn_T3(T3,F,o,v))

    print("Estimated difference between T3^fnT3 - (T2^WT3 + h.c.) is:",total_FO_trples,total_FO_trples**2)
    uccsd_fifthorder_corrections(W,T3,o,v,D3)
    variance = total_FO_trples**2
    return total_FO_trples, variance

def uccsd_fifthorder_corrections(W,T3SO,o,v,D3):
    import pycc.build_sqrbrak_corrections as build_sqrbrak_corrections
    D3T3 = pycc.build_sqrbrak_corrections.buildTO_WT3_to_T3(W,o,v,T3SO)
    D3T3 = pycc.tamps.antisym_T3(D3T3,None,None)
    E5_t3SOdag_wnT3SO = 0.25* pycc.build_sqrbrak_corrections.sqr_brakT_spin(D3T3,T3SO.transpose(3,4,5,0,1,2))
    print("Fifth-order contribution to this:",E5_t3SOdag_wnT3SO)
    return E5_t3SOdag_wnT3SO
