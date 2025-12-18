import numpy as np
import pycc.tamps as tamps
import pycc.build_sqrbrak_corrections as build_sqrbrak_corrections

def build_eom_sqrbrakT_resid(W,T1,T2,C2,o,v):
    # return antisymmetrized terms defininig D3C3

    D3C3_WT1C2 = build_Q3_WT1C2(W,T1,C2,o,v)
    D3C3_WC2   = build_Q3_WnC2(W,C2,o,v)

    return D3C3_WT1C2, D3C3_WC2


def build_Q3_WT1C2(W,T1,C2,o,v):
    no = np.shape(T1)[0]
    nv = np.shape(T1)[1]
    d3c3 = np.zeros((no,no,no,nv,nv,nv))
    d3c3 += 0.250000000 * np.einsum("ilab,mc,jklm->ijkabc",C2,T1,W[o,o,o,o],optimize="optimal")
    d3c3 += 0.500000000 * np.einsum("ilab,jd,kdlc->ijkabc",C2,T1,W[o,v,o,v],optimize="optimal")
    d3c3 += 0.500000000 * np.einsum("ijad,lb,kdlc->ijkabc",C2,T1,W[o,v,o,v],optimize="optimal")
    d3c3 += 0.250000000 * np.einsum("ijad,ke,debc->ijkabc",C2,T1,W[v,v,v,v],optimize="optimal")

    fin_d3c3 = tamps.antisym_T3(d3c3, no, nv)
    return fin_d3c3


def build_Q3_WnC2(W,C2,o,v):
    no = np.shape(C2)[0]
    nv = np.shape(C2)[2]
    d3c3 = build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,C2)
    fin_d3c3 = tamps.antisym_T3(d3c3,no,nv)
    return fin_d3c3


