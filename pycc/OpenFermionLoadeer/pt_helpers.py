import pycc
import numpy as np

def uccsd_FO_triples_corrections(F,W,T2,o,v,D3):
    D3T3 = pycc.build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,T2)
    D3T3 = pycc.tamps.antisym_T3(D3T3,None,None)
    T3 = D3T3*D3 #.transpose(3,4,5,0,1,2)
    sqrBrak_T =0.25* pycc.build_sqrbrak_corrections.sqr_brakT_spin(D3T3,T3.transpose(3,4,5,0,1,2))

    total_FO_trples = 2.0*sqrBrak_T -  pycc.build_sqrbrak_corrections.build_T3dag_fn_T3(T3,F,o,v)
    print("Estimated difference between T3^fnT3 - (T2^WT3 + h.c.) is:",total_FO_trples,total_FO_trples**2)
    return total_FO_trples

