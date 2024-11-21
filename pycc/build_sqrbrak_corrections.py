import numpy as np
import pycc.tamps as tamps

def build_FOsqrBrakTriples(): 
    t3resid=wicked_T3corr.build_T3_secondO(self.g,self.o,self.v,self.t2)
    t3resid=wicked_T3corr.antisym_T3(t3resid,self.nocc,self.nvirt)
    t3residOrigContract=t3resid
    t3resid=t3resid.transpose(3,4,5,0,1,2)
    t3=t3resid*self.denoms["D3aa"]
    t3Contract=t3
    t3=t3.transpose(3,4,5,0,1,2)

    energy=wicked_T3corr.getE_squareBrackT(self.g,self.o,self.v,t3,t2_dag)
    print('[T]-based triples energy correction to CC is:',energy)
    return energy

def build_FOsqrBrakSingles():
    netT1=get_netT1_fromT2(self.g,self.o,self.v,self.t2)
    t1_bar=netT1.transpose(1,0)*self.denoms["D1aa"]
    print('shapes',np.shape(netT1),np.shape(t1_bar))
    print('check on [S]',np.einsum("ia,ai->",t1_bar,netT1))
    t1_bar=t1_bar.transpose(1,0)
    #t2_bar_resid=get_netT2_fromT1bar(self.g,self.o,self.v,t1_bar)
    D2T2_aa,D2T2_bb,D2T2_ab = get_netT2_fromT1bar(T1_aa,T1_bb,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)

    D2T2_aa = tamps.antisym_T2(D2T2_aa,None,None)
    D2T2_bb = tamps.antisym_T2(D2T2_bb,None,None)
    D2T2_ab = tamps.antisym_T2(D2T2_ab,None,None)

    #t2_bar_resid=wicked_T3corr.antisym_T2(t2_bar_resid,self.nocc,self.nvirt)
    #singles_correction=0.250000000 * np.einsum("abij,ijab->",t2_dag,t2_bar_resid,optimize="optimal")
    singles_correction = get_singles_correction(T2_aa,T2_bb,T2_ab,D2T2dag_aa,D2T2dag_bb,D2T2dag_ab)
    print('[S] singles correction to UCC:',singles_correction)
    return singles_correction





def get_netT1_fromT2():
    Rov = np.zeros((nocc,nvir))
    Rov += -0.500000000 * np.einsum("ibjk,jkab->ia",W_aaaa[oa,va,oa,oa],T2_aa,optimize="optimal")
    Rov += -0.500000000 * np.einsum("bcja,ijbc->ia",W_aaaa[va,va,oa,va],T2_aa,optimize="optimal")
    Rov += -1.000000000 * np.einsum("iAjI,jIaA->ia",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    Rov += 1.000000000 * np.einsum("bAaI,iIbA->ia",W_abab[va,vb,va,ob],T2_ab,optimize="optimal")

    ROV = np.zeros((nocc,nvir))
    ROV += -1.000000000 * np.einsum("aIiJ,iJaA->IA",W_abab[va,ob,oa,ob],T2_ab,optimize="optimal")
    ROV += 1.000000000 * np.einsum("aBiA,iIaB->IA",W_abab[va,vb,oa,ob],T2_ab,optimize="optimal")
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
