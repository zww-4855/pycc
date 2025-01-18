import numpy as np
import pycc.tamps as tamps

def build_T1_fromT2_SOspin(g,o,v,t2):
    rov = -0.500000000 * np.einsum("jkab,ibjk->ia",t2,g[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ijbc,bcja->ia",t2,g[v,v,o,v],optimize="optimal")
    return rov

def sqr_brakT_spin(t3resid,t3_dag):
    return 0.111111111 * np.einsum("ijkabc,abcijk->",t3resid,t3_dag,optimize="optimal")

def build_T3_secondO_spin(g,o,v,t2):
    rooovvv = -0.250000000 * np.einsum("ilab,jklc->ijkabc",t2,g[o,o,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",t2,g[o,v,v,v],optimize="optimal")
    return rooovvv


## ADDED ZWW 1/18/2025 for fifth-order triples' corrections
def buildTO_WT3_to_T3(W,o,v,T3):
    rooovvv = 0.041666667 * np.einsum("ilmabc,jklm->ijkabc",T3,W[o,o,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijlabd,kdlc->ijkabc",T3,W[o,v,o,v],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("ijkade,debc->ijkabc",T3,W[v,v,v,v],optimize="optimal")
    return rooovvv

def buildTO_wnT2sqr_to_T3(W,o,v,T2):
    rooovvv = 0.500000000 * np.einsum("ilab,jmcd,kdlm->ijkabc",T2,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("ilab,jkde,delc->ijkabc",T2,T2,W[v,v,o,v],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("lmab,ijcd,kdlm->ijkabc",T2,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("ijad,klbe,delc->ijkabc",T2,T2,W[v,v,o,v],optimize="optimal")
    return rooovvv


def buildTO_wnT1T2_to_T3(W,o,v,T1,T2):
    rooovvv = -0.250000000 * np.einsum("ijlm,la,kmbc->ijkabc",W[o,o,o,o],T1,T2,optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("idla,lb,jkcd->ijkabc",W[o,v,o,v],T1,T2,optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("idla,jd,klbc->ijkabc",W[o,v,o,v],T1,T2,optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("deab,id,jkce->ijkabc",W[v,v,v,v],T1,T2,optimize="optimal")
    return rooovvv

def build_FOsqrBrakTriples(driveCCobj,T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):

    Rooovvv,RooOvvV, RoOOvVV, ROOOVVV = get_netT3_fromT2(T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)

    nocc=np.shape(Rooovvv)[0]
    nvir=np.shape(Rooovvv)[3]
    Rooovvv = tamps.antisym_T3SI_aaa(Rooovvv, nocc, nvir)
    ROOOVVV = tamps.antisym_T3SI_aaa(ROOOVVV, nocc, nvir)
    RooOvvV = tamps.antisym_T3SI_aab(RooOvvV, nocc, nvir)
    RoOOvVV = tamps.antisym_T3SI_abb(RoOOvVV, nocc, nvir)

    T3dag_aaa = (Rooovvv*driveCCobj.denomInfo["D3aaa"]).transpose(3,4,5,0,1,2)
    T3dag_bbb = (ROOOVVV*driveCCobj.denomInfo["D3bbb"]).transpose(3,4,5,0,1,2)
    T3dag_aab = (RooOvvV*driveCCobj.denomInfo["D3aab"]).transpose(3,4,5,0,1,2)
    T3dag_abb = (RoOOvVV*driveCCobj.denomInfo["D3abb"]).transpose(3,4,5,0,1,2)



    triples_correction = 0.111111111 * np.einsum("ijkabc,abcijk->",Rooovvv,T3dag_aaa,optimize="optimal")
    triples_correction += 0.250000000 * np.einsum("ijIabA,abAijI->",RooOvvV,T3dag_aab,optimize="optimal")
    triples_correction += 0.250000000 * np.einsum("iIJaAB,aABiIJ->",RoOOvVV,T3dag_abb,optimize="optimal")
    triples_correction += 0.111111111 * np.einsum("IJKABC,ABCIJK->",ROOOVVV,T3dag_bbb,optimize="optimal")

    print('[T] correction to UCC is:', 0.250000000 * np.einsum("ijIabA,abAijI->",RooOvvV,T3dag_aab,optimize="optimal"))
    print('other addon',0.250000000 * np.einsum("iIJaAB,aABiIJ->",RoOOvVV,T3dag_abb,optimize="optimal"))
    print('last:', 0.111111111 * np.einsum("IJKABC,ABCIJK->",ROOOVVV,T3dag_bbb,optimize="optimal"))
    #sys.exit()
    return triples_correction

def build_FOsqrBrakSingles(driveCCobj,T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    #netT1=get_netT1_fromT2(self.g,self.o,self.v,self.t2)
    netT1_aa, netT1_bb = get_netT1_fromT2(T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)

    t1bar_aa = netT1_aa*driveCCobj.denomInfo["D1aa"]
    t1bar_bb = netT1_bb*driveCCobj.denomInfo["D1bb"]

    singles_correction = 1.000000000 * np.einsum("ia,ai->",t1bar_aa,netT1_aa.transpose(1,0),optimize="optimal")
    singles_correction += 1.000000000 * np.einsum("IA,AI->",t1bar_bb,netT1_bb.transpose(1,0),optimize="optimal")

    print('[S] singles correction to UCC:',singles_correction)
    return singles_correction




def get_netT3_fromT2(T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    Rooovvv = -0.250000000 * np.einsum("ijla,klbc->ijkabc",W_aaaa[oa,oa,oa,va],T2_aa,optimize="optimal")
    Rooovvv += -0.250000000 * np.einsum("idab,jkcd->ijkabc",W_aaaa[oa,va,va,va],T2_aa,optimize="optimal")


    RooOvvV = 0.500000000 * np.einsum("ijka,kIbA->ijIabA",W_aaaa[oa,oa,oa,va],T2_ab,optimize="optimal")
    RooOvvV += 0.500000000 * np.einsum("iIkA,jkab->ijIabA",W_abab[oa,ob,oa,vb],T2_aa,optimize="optimal")
    RooOvvV += 0.500000000 * np.einsum("icab,jIcA->ijIabA",W_aaaa[oa,va,va,va],T2_ab,optimize="optimal")
    RooOvvV += -1.000000000 * np.einsum("iIaJ,jJbA->ijIabA",W_abab[oa,ob,va,ob],T2_ab,optimize="optimal")
    RooOvvV += 1.000000000 * np.einsum("iBaA,jIbB->ijIabA",W_abab[oa,vb,va,vb],T2_ab,optimize="optimal")
    RooOvvV += -0.500000000 * np.einsum("cIaA,ijbc->ijIabA",W_abab[va,ob,va,vb],T2_aa,optimize="optimal")


    RoOOvVV = -1.000000000 * np.einsum("iIjA,jJaB->iIJaAB",W_abab[oa,ob,oa,vb],T2_ab,optimize="optimal")
    RoOOvVV += 0.500000000 * np.einsum("iIaK,JKAB->iIJaAB",W_abab[oa,ob,va,ob],T2_bb,optimize="optimal")
    RoOOvVV += -0.500000000 * np.einsum("iCaA,IJBC->iIJaAB",W_abab[oa,vb,va,vb],T2_bb,optimize="optimal")
    RoOOvVV += 1.000000000 * np.einsum("bIaA,iJbB->iIJaAB",W_abab[va,ob,va,vb],T2_ab,optimize="optimal")
    RoOOvVV += 0.500000000 * np.einsum("IJKA,iKaB->iIJaAB",W_bbbb[ob,ob,ob,vb],T2_ab,optimize="optimal")
    RoOOvVV += 0.500000000 * np.einsum("ICAB,iJaC->iIJaAB",W_bbbb[ob,vb,vb,vb],T2_ab,optimize="optimal")


    ROOOVVV = -0.250000000 * np.einsum("IJLA,KLBC->IJKABC",W_bbbb[ob,ob,ob,vb],T2_bb,optimize="optimal")
    ROOOVVV += -0.250000000 * np.einsum("IDAB,JKCD->IJKABC",W_bbbb[ob,vb,vb,vb],T2_bb,optimize="optimal")


    return Rooovvv,RooOvvV, RoOOvVV, ROOOVVV



def get_netT1_fromT2(T2_aa,T2_bb,T2_ab,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    Rov = -0.500000000 * np.einsum("ibjk,jkab->ia",W_aaaa[oa,va,oa,oa],T2_aa,optimize="optimal")
    Rov += -0.500000000 * np.einsum("bcja,ijbc->ia",W_aaaa[va,va,oa,va],T2_aa,optimize="optimal")
    Rov += -1.000000000 * np.einsum("iAjI,jIaA->ia",W_abab[oa,vb,oa,ob],T2_ab,optimize="optimal")
    Rov += 1.000000000 * np.einsum("bAaI,iIbA->ia",W_abab[va,vb,va,ob],T2_ab,optimize="optimal")

    ROV = -1.000000000 * np.einsum("aIiJ,iJaA->IA",W_abab[va,ob,oa,ob],T2_ab,optimize="optimal")
    ROV += 1.000000000 * np.einsum("aBiA,iIaB->IA",W_abab[va,vb,oa,vb],T2_ab,optimize="optimal")
    ROV += -0.500000000 * np.einsum("IBJK,JKAB->IA",W_bbbb[ob,vb,ob,ob],T2_bb,optimize="optimal")
    ROV += -0.500000000 * np.einsum("BCJA,IJBC->IA",W_bbbb[vb,vb,ob,vb],T2_bb,optimize="optimal")

    return Rov, ROV


def get_netT2_fromT1bar(T1_aa,T1_bb,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    Roovv = 0.500000000 * np.einsum("ijka,kb->ijab",W_aaaa[oa,oa,oa,va],T1_aa,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("icab,jc->ijab",W_aaaa[oa,va,va,va],T1_aa,optimize="optimal")

    ROOVV = 0.500000000 * np.einsum("IJKA,KB->IJAB",W_bbbb[ob,ob,ob,vb],T1_bb,optimize="optimal")
    ROOVV += 0.500000000 * np.einsum("ICAB,JC->IJAB",W_bbbb[ob,vb,vb,vb],T1_bb,optimize="optimal")

    RoOvV = -1.000000000 * np.einsum("iIjA,ja->iIaA",W_abab[oa,ob,oa,vb],T1_aa,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iIaJ,JA->iIaA",W_abab[oa,ob,va,ob],T1_bb,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iBaA,IB->iIaA",W_abab[oa,vb,va,vb],T1_bb,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bIaA,ib->iIaA",W_abab[va,ob,va,vb],T1_aa,optimize="optimal")

    return Roovv, ROOVV, RoOvV


def get_singles_correction(T2_aa,T2_bb,T2_ab,D2T2dag_aa,D2T2dag_bb,D2T2dag_ab):

    R = 0.250000000 * np.einsum("abij,ijab->",D2T2dag_aa,T2_aa,optimize="optimal")
    R += 1.000000000 * np.einsum("aAiI,iIaA->",D2T2dag_ab,T2_ab,optimize="optimal")
    R += 0.250000000 * np.einsum("ABIJ,IJAB->",D2T2dag_bb,T2_bb,optimize="optimal")



    R1 = 0.250000000 * np.einsum("abij,ijab->",D2T2dag_aa,T2_aa,optimize="optimal")
    R1 += 1.000000000 * np.einsum("aAiI,iIaA->",D2T2dag_ab,T2_ab,optimize="optimal")
    R1 += 0.250000000 * np.einsum("ABIJ,IJAB->",D2T2dag_bb,T2_bb,optimize="optimal")



    R2 = 0.250000000 * np.einsum("abij,ijab->",D2T2dag_aa,T2_aa,optimize="optimal")
    R2 += 1.000000000 * np.einsum("aAiI,iIaA->",D2T2dag_ab,T2_ab,optimize="optimal")
    R2 += 0.250000000 * np.einsum("ABIJ,IJAB->",D2T2dag_bb,T2_bb,optimize="optimal")

    print('alpha,beta,mixed contribution to [S]:',R,R1,R2)
    return R+R1+R2
