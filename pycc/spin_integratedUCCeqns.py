import numpy as np
import pycc.tamps as tamps



def UCC3_t2resid_aa(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock):
    #Roovv = np.zeros((oa,oa,va,va))
    Roovv = 0.500000000 * np.einsum("ik,jkab->ijab",Fock[oa,oa],T2_aa,optimize="optimal")
    Roovv += 0.125000000 * np.einsum("ijkl,klab->ijab",W_aaaa[oa,oa,oa,oa],T2_aa,optimize="optimal")
    Roovv += 0.250000000 * np.einsum("cdkl,ilab,jkcd->ijab",W_aaaa[va,va,oa,oa],T2_aa,T2_aa,optimize="optimal")
    Roovv += 0.062500000 * np.einsum("cdkl,klab,ijcd->ijab",W_aaaa[va,va,oa,oa],T2_aa,T2_aa,optimize="optimal")
    Roovv += 0.250000000 * np.einsum("cdkl,ijad,klbc->ijab",W_aaaa[va,va,oa,oa],T2_aa,T2_aa,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("cdkl,ilad,jkbc->ijab",W_aaaa[va,va,oa,oa],T2_aa,T2_aa,optimize="optimal")
    Roovv += -1.000000000 * np.einsum("icka,jkbc->ijab",W_aaaa[oa,va,oa,va],T2_aa,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("cAkI,iIaA,jkbc->ijab",W_abab[va,vb,oa,ob],T2_ab,T2_aa,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("cAkI,kIaA,ijbc->ijab",W_abab[va,vb,oa,ob],T2_ab,T2_aa,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("cAkI,iIcA,jkab->ijab",W_abab[va,vb,oa,ob],T2_ab,T2_aa,optimize="optimal")
    Roovv += -0.500000000 * np.einsum("ca,ijbc->ijab",Fock[va,va],T2_aa,optimize="optimal")
    Roovv += 0.250000000 * np.einsum("ijab->ijab",W_aaaa[oa,oa,va,va],optimize="optimal")
    Roovv += 0.125000000 * np.einsum("cdab,ijcd->ijab",W_aaaa[va,va,va,va],T2_aa,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("iAaI,jIbA->ijab",W_abab[oa,vb,va,ob],T2_ab,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("ABIJ,iIaA,jJbB->ijab",W_bbbb[vb,vb,ob,ob],T2_ab,T2_ab,optimize="optimal")
    nocc = nvir = None
    Roovv = tamps.antisym_T2(Roovv,nocc,nvir)

    return Roovv



def UCC3_t2resid_bb(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock):
    #ROOVV = np.zeros((ob,ob,vb,vb))
    ROOVV = 0.500000000 * np.einsum("abij,jIbA,iJaB->IJAB",W_aaaa[va,va,oa,oa],T2_ab,T2_ab,optimize="optimal")
    ROOVV += -0.500000000 * np.einsum("aCiK,IKAB,iJaC->IJAB",W_abab[va,vb,oa,ob],T2_bb,T2_ab,optimize="optimal")
    ROOVV += -0.500000000 * np.einsum("aCiK,IJAC,iKaB->IJAB",W_abab[va,vb,oa,ob],T2_bb,T2_ab,optimize="optimal")
    ROOVV += 1.000000000 * np.einsum("aCiK,IKAC,iJaB->IJAB",W_abab[va,vb,oa,ob],T2_bb,T2_ab,optimize="optimal")
    ROOVV += 1.000000000 * np.einsum("aIiA,iJaB->IJAB",W_abab[va,ob,oa,vb],T2_ab,optimize="optimal")
    ROOVV += 0.500000000 * np.einsum("IK,JKAB->IJAB",Fock[ob,ob],T2_bb,optimize="optimal")
    ROOVV += 0.125000000 * np.einsum("IJKL,KLAB->IJAB",W_bbbb[ob,ob,ob,ob],T2_bb,optimize="optimal")
    ROOVV += 0.250000000 * np.einsum("CDKL,ILAB,JKCD->IJAB",W_bbbb[vb,vb,ob,ob],T2_bb,T2_bb,optimize="optimal")
    ROOVV += 0.062500000 * np.einsum("CDKL,KLAB,IJCD->IJAB",W_bbbb[vb,vb,ob,ob],T2_bb,T2_bb,optimize="optimal")
    ROOVV += 0.250000000 * np.einsum("CDKL,IJAD,KLBC->IJAB",W_bbbb[vb,vb,ob,ob],T2_bb,T2_bb,optimize="optimal")
    ROOVV += 0.500000000 * np.einsum("CDKL,ILAD,JKBC->IJAB",W_bbbb[vb,vb,ob,ob],T2_bb,T2_bb,optimize="optimal")
    ROOVV += -1.000000000 * np.einsum("ICKA,JKBC->IJAB",W_bbbb[ob,vb,ob,vb],T2_bb,optimize="optimal")
    ROOVV += -0.500000000 * np.einsum("CA,IJBC->IJAB",Fock[vb,vb],T2_bb,optimize="optimal")
    ROOVV += 0.250000000 * np.einsum("IJAB->IJAB",W_bbbb[ob,ob,vb,vb],optimize="optimal")
    ROOVV += 0.125000000 * np.einsum("CDAB,IJCD->IJAB",W_bbbb[vb,vb,vb,vb],T2_bb,optimize="optimal")
    nocc = nvir = None
    ROOVV = tamps.antisym_T2(ROOVV,nocc,nvir)

    return ROOVV


def UCC3_t2resid_ab(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock_aa,Fock_bb):
    #RoOvV = np.zeros((oa,ob,va,vb))
    RoOvV = -1.000000000 * np.einsum("ij,jIaA->iIaA",Fock_aa[oa,oa],T2_ab,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("bcjk,jIaA,ikbc->iIaA",W_aaaa[va,va,oa,oa],T2_ab,T2_aa,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("bcjk,iIbA,jkac->iIaA",W_aaaa[va,va,oa,oa],T2_ab,T2_aa,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bcjk,jIbA,ikac->iIaA",W_aaaa[va,va,oa,oa],T2_ab,T2_aa,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("ibja,jIbA->iIaA",W_aaaa[oa,va,oa,va],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIjJ,jJaA->iIaA",W_abab[oa,ob,oa,ob],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bBjJ,iJaA,jIbB->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bBjJ,jIaA,iJbB->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBjJ,jJaA,iIbB->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bBjJ,iIaB,jJbA->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBjJ,iJaB,jIbA->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBjJ,jIaB,iJbA->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bBjJ,jJaB,iIbA->iIaA",W_abab[va,vb,oa,ob],T2_ab,T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBjJ,IJAB,ijab->iIaA",W_abab[va,vb,oa,ob],T2_bb,T2_aa,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iBjA,jIaB->iIaA",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bIjA,ijab->iIaA",W_abab[va,ob,oa,vb],T2_aa,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("ba,iIbA->iIaA",Fock_aa[va,va],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iBaJ,IJAB->iIaA",W_abab[oa,vb,va,ob],T2_bb,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bIaJ,iJbA->iIaA",W_abab[va,ob,va,ob],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIaA->iIaA",W_abab[oa,ob,va,vb],optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBaA,iIbB->iIaA",W_abab[va,vb,va,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("IJ,iJaA->iIaA",Fock_bb[ob,ob],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("BCJK,IJAB,iKaC->iIaA",W_bbbb[vb,vb,ob,ob],T2_bb,T2_ab,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("BCJK,JKAB,iIaC->iIaA",W_bbbb[vb,vb,ob,ob],T2_bb,T2_ab,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("BCJK,IJBC,iKaA->iIaA",W_bbbb[vb,vb,ob,ob],T2_bb,T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("IBJA,iJaB->iIaA",W_bbbb[ob,vb,ob,vb],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("BA,iIaB->iIaA",Fock_bb[vb,vb],T2_ab,optimize="optimal")
    #nocc = nvir = None
    #RoOvV = tamps.antisym_T2(RoOvV,nocc,nvir)


    return RoOvV


