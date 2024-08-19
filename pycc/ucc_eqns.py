import numpy as np
import pycc.tamps as tamps

def ucc_eqnDriver(calcType,Fock,W,T1,T2,o,v):
    if calcType == "UCCD3":
        D2T2 = ucc3_t2resid(Fock,W,T2,o,v)
#        return D2T2

    if calcType == "UCCSD4" or calcType == "UCCD4":
        D1T1 = uccsd4_t1resid(Fock,W,T1,T2,o,v)
        D2T2 = uccsd4_t2resid(Fock,W,T1,T2,o,v)
#        return D1T1, D2T2
        
    if calcType == "UCCSD5" or calcType == "UCCD5":
        D1T1 = uccsd4_t1resid(Fock,W,T1,T2,o,v)
        D1T1 += uccsd5_t1resid_linearTerms(W,T1,o,v)
        D1T1 += uccsd5_t1resid_quadTerm(W,T2,o,v)


        D2T2 =  uccsd4_t2resid(Fock,W,T1,T2,o,v)
        D2T2 += uccsd5_t2resid_T1couplings(W,T1,T2,o,v)


    nocc=nvir=None
    D2T2=tamps.antisym_T2(D2T2,nocc,nvir)
    return D1T1,D2T2

def ucc3_t2resid(Fock,W,T2,o,v):
    roovv = 0.500000000 * np.einsum("ik,jkab->ijab",Fock[o,o],T2,optimize="optimal")
    roovv += -0.500000000 * np.einsum("ca,ijbc->ijab",Fock[v,v],T2,optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")
    return roovv

def uccsd4_t1resid(Fock,W,T1,T2,o,v):
    rov = -1.000000000 * np.einsum("ij,ja->ia",Fock[o,o],T1,optimize="optimal")
    rov += 1.000000000 * np.einsum("ba,ib->ia",Fock[v,v],T1,optimize="optimal")
    rov += -0.500000000 * np.einsum("jkab,ibjk->ia",T2,W[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ijbc,bcja->ia",T2,W[v,v,o,v],optimize="optimal")
    return rov

def uccsd4_t2resid(Fock,W,T1,T2,o,v):
    # terms linear in T
    roovv = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += ucc3_t2resid(Fock,W,T2,o,v)

    # terms quadratic in T2
    tmp_wnT2sqr=0.5*uccsd_wnT2sqr(W,T2,o,v)
    roovv += tmp_wnT2sqr

    # T2dagWnT2, Q2[[W,tau2],tau2], or 0.5*[[T2dag,W],T2]
    roovv += 0.5*uccsd_T2dagWnT2(W,T2,o,v)
#    nocc=nvir=None
#    roovv=tamps.antisym_T2(roovv,nocc,nvir)
    return roovv






def uccsd_wnT2sqr(W,T2,o,v):
    roovv = -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    return roovv

def uccsd_T2dagWnT2(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ikab,cdkl,jlcd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,cdkl,ijcd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,cdkl,klbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,cdkl,jlbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,cdkl,ijbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("ijcd,cdkl,klab->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikcd,cdkl,jlab->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")

    return roovv


def uccsd5_t1resid_linearTerms(W,T1,o,v):

    # Q1(WnT1)|0>
    rov = -1.000000000 * np.einsum("jb,ibja->ia",T1,W[o,v,o,v],optimize="optimal")
    # Q1(T1^W)|0>
    rov += 1.000000000 * np.einsum("bj,ijab->ia",T1dag,W[o,o,v,v],optimize="optimal")
    return rov

def uccsd5_t1resid_quadTerm(W,T2,o,v):
    # Q1(T2^WnT2)|0>
    T2dag=T2.transpose(2,3,0,1)
    rov = -0.500000000 * np.einsum("ijab,cdjk,kbcd->ia",T2,T2dag,W[o,v,v,v],optimize="optimal")
    rov += -0.500000000 * np.einsum("ijab,bckl,kljc->ia",T2,T2dag,W[o,o,o,v],optimize="optimal")
    rov += -0.250000000 * np.einsum("jkab,cdjk,ibcd->ia",T2,T2dag,W[o,v,v,v],optimize="optimal")
    rov += 1.000000000 * np.einsum("jkab,bcjl,ilkc->ia",T2,T2dag,W[o,o,o,v],optimize="optimal")
    rov += 1.000000000 * np.einsum("ijbc,bdjk,kcad->ia",T2,T2dag,W[o,v,v,v],optimize="optimal")
    rov += -0.250000000 * np.einsum("ijbc,bckl,klja->ia",T2,T2dag,W[o,o,o,v],optimize="optimal")
    rov += 0.500000000 * np.einsum("jkbc,bdjk,icad->ia",T2,T2dag,W[o,v,v,v],optimize="optimal")
    rov += 0.500000000 * np.einsum("jkbc,bcjl,ilka->ia",T2,T2dag,W[o,o,o,v],optimize="optimal")
    return rov

def uccsd5_t2resid_T1couplings(W,T1,T2,o,v):
    T1dag = T1.transpose(1,0)

    # WnT1T2
    roovv = 1.000000000 * np.einsum("ka,ilbc,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ka,ijcd,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ic,klab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ic,jkad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ilab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ijad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")

    # T1^WnT2
    roovv += -0.500000000 * np.einsum("ck,ilab,jklc->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ck,klab,ijlc->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ck,ilac,jklb->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ck,ijad,kdbc->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ck,ikad,jdbc->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ck,ijcd,kdab->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    return roovv
