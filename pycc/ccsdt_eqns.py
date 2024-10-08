import numpy as np
import pycc.tamps as tamps
import pycc.build_qf_correction as qf
import pycc.modify_T2resid_T4Qf1Slow as pdag_xcc5
import pycc.modify_T2resid_T4Qf2Slow as pdag_xcc6


def ccsdt_t1eqns(F,W,T1,T2,T3,o,v):
    # contributions to the residual
    rov = -1.000000000 * np.einsum("ij,ja->ia",F[o,o],T1,optimize="optimal")
    rov += -1.000000000 * np.einsum("bj,ja,ib->ia",F[v,o],T1,T1,optimize="optimal")
    rov += 1.000000000 * np.einsum("bj,ijab->ia",F[v,o],T2,optimize="optimal")
    rov += 1.000000000 * np.einsum("ia->ia",F[o,v],optimize="optimal")
    rov += 1.000000000 * np.einsum("ba,ib->ia",F[v,v],T1,optimize="optimal")
    rov += -1.000000000 * np.einsum("ja,ib,kc,bcjk->ia",T1,T1,T1,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("ja,kb,ibjk->ia",T1,T1,W[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ja,ikbc,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("ib,jc,bcja->ia",T1,T1,W[v,v,o,v],optimize="optimal")
    rov += -0.500000000 * np.einsum("ib,jkac,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += 1.000000000 * np.einsum("jb,ikac,bcjk->ia",T1,T2,W[v,v,o,o],optimize="optimal")
    rov += -1.000000000 * np.einsum("jb,ibja->ia",T1,W[o,v,o,v],optimize="optimal")
    rov += -0.500000000 * np.einsum("jkab,ibjk->ia",T2,W[o,v,o,o],optimize="optimal")
    rov += -0.500000000 * np.einsum("ijbc,bcja->ia",T2,W[v,v,o,v],optimize="optimal")
    rov += 0.250000000 * np.einsum("ijkabc,bcjk->ia",T3,W[v,v,o,o],optimize="optimal")
    return rov

def ccsdt_t2eqns(F,W,T1,T2,T3,o,v,CCobj):
    # contributions to the residual
    roovv = 0.500000000 * np.einsum("ik,jkab->ijab",F[o,o],T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ka,ijbc->ijab",F[v,o],T1,T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ic,jkab->ijab",F[v,o],T1,T2,optimize="optimal")
    roovv += 0.250000000 * np.einsum("ck,ijkabc->ijab",F[v,o],T3,optimize="optimal")
    roovv += -0.500000000 * np.einsum("ca,ijbc->ijab",F[v,v],T2,optimize="optimal")
    roovv += 0.250000000 * np.einsum("ka,lb,ic,jd,cdkl->ijab",T1,T1,T1,T1,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,lb,ic,jckl->ijab",T1,T1,T1,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ka,lb,ijcd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ka,lb,ijkl->ijab",T1,T1,W[o,o,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,ic,jd,cdkb->ijab",T1,T1,T1,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ka,ic,jlbd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ka,ic,jckb->ijab",T1,T1,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,lc,ijbd,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ka,ilbc,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ka,ijcd,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ka,ijlbcd,cdkl->ijab",T1,T3,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ic,jd,klab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ic,jd,cdab->ijab",T1,T1,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,kd,jlab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ic,klab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ic,jkad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ic,jklabd,cdkl->ijab",T1,T3,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ilab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ijad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("kc,ijlabd,cdkl->ijab",T1,T3,W[v,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("iklabc,jckl->ijab",T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijkacd,cdkb->ijab",T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")

    nocc=nvir=None
    roovv=tamps.antisym_T2(roovv,nocc,nvir)

    if 'qf-' in CCobj.cc_type: 
        D2T2 = 0.5*pdag_xcc5.residQf1_aaaa(W,T2,T2.transpose(2,3,0,1),o,v)
        D2T2 += 0.5*qf.wnT3_pdagq(W,T2,T3.transpose(3,4,5,0,1,2),o,v)
        print('inside qf-1')
        if 'qf-2' in CCobj.cc_type:
            print('inside qf-2')
            D2T2 += 0.5*pdag_xcc6.residQf2_aaaa(W,T2,T2.transpose(2,3,0,1),o,v)
            D2T2 += qf.wnT2T3_pdagq(W,T2,T2.transpose(2,3,0,1),T3.transpose(3,4,5,0,1,2),o,v)

        D2T2 = D2T2.transpose(2,3,0,1)
        roovv += D2T2
    return roovv


def ccsdt_t3eqns(F,W,T1,T2,T3,o,v):
    # contributions to the residual
    rooovvv = -0.083333333 * np.einsum("il,jklabc->ijkabc",F[o,o],T3,optimize="optimal")
    rooovvv += -0.083333333 * np.einsum("dl,la,ijkbcd->ijkabc",F[v,o],T1,T3,optimize="optimal")
    rooovvv += -0.083333333 * np.einsum("dl,id,jklabc->ijkabc",F[v,o],T1,T3,optimize="optimal")
    rooovvv += 0.250000000 * np.einsum("dl,ilab,jkcd->ijkabc",F[v,o],T2,T2,optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("da,ijkbcd->ijkabc",F[v,v],T3,optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,mb,id,jkce,delm->ijkabc",T1,T1,T1,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,mb,ijcd,kdlm->ijkabc",T1,T1,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("la,mb,ijkcde,delm->ijkabc",T1,T1,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,id,je,kmbc,delm->ijkabc",T1,T1,T1,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("la,id,jmbc,kdlm->ijkabc",T1,T1,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("la,id,jkbe,delc->ijkabc",T1,T1,T2,W[v,v,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,id,jkmbce,delm->ijkabc",T1,T1,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("la,md,ijkbce,delm->ijkabc",T1,T1,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("la,imbc,jkde,delm->ijkabc",T1,T2,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,imbc,jklm->ijkabc",T1,T2,W[o,o,o,o],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("la,ijbd,kmce,delm->ijkabc",T1,T2,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("la,ijbd,kdlc->ijkabc",T1,T2,W[o,v,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("la,ijmbcd,kdlm->ijkabc",T1,T3,W[o,v,o,o],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("la,ijkbde,delc->ijkabc",T1,T3,W[v,v,o,v],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("id,je,klab,delc->ijkabc",T1,T1,T2,W[v,v,o,v],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("id,je,klmabc,delm->ijkabc",T1,T1,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("id,le,jkmabc,delm->ijkabc",T1,T1,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("id,jlab,kmce,delm->ijkabc",T1,T2,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.500000000 * np.einsum("id,jlab,kdlc->ijkabc",T1,T2,W[o,v,o,v],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("id,lmab,jkce,delm->ijkabc",T1,T2,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("id,jkae,debc->ijkabc",T1,T2,W[v,v,v,v],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("id,jlmabc,kdlm->ijkabc",T1,T3,W[o,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("id,jklabe,delc->ijkabc",T1,T3,W[v,v,o,v],optimize="optimal")
    rooovvv += 0.250000000 * np.einsum("ld,imab,jkce,delm->ijkabc",T1,T2,T2,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("ld,ijmabc,kdlm->ijkabc",T1,T3,W[o,v,o,o],optimize="optimal")
    rooovvv += 0.083333333 * np.einsum("ld,ijkabe,delc->ijkabc",T1,T3,W[v,v,o,v],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("ilab,jmcd,kdlm->ijkabc",T2,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("ilab,jkde,delc->ijkabc",T2,T2,W[v,v,o,v],optimize="optimal")
    rooovvv += 0.125000000 * np.einsum("ilab,jkmcde,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ilab,jklc->ijkabc",T2,W[o,o,o,v],optimize="optimal")
    rooovvv += -0.125000000 * np.einsum("lmab,ijcd,kdlm->ijkabc",T2,T2,W[o,v,o,o],optimize="optimal")
    rooovvv += 0.020833333 * np.einsum("lmab,ijkcde,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.500000000 * np.einsum("ijad,klbe,delc->ijkabc",T2,T2,W[v,v,o,v],optimize="optimal")
    rooovvv += 0.125000000 * np.einsum("ijad,klmbce,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijad,kdbc->ijkabc",T2,W[o,v,v,v],optimize="optimal")
    rooovvv += 0.250000000 * np.einsum("ilad,jkmbce,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("lmad,ijkbce,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.020833333 * np.einsum("ijde,klmabc,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("ilde,jkmabc,delm->ijkabc",T2,T3,W[v,v,o,o],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("ilmabc,jklm->ijkabc",T3,W[o,o,o,o],optimize="optimal")
    rooovvv += -0.250000000 * np.einsum("ijlabd,kdlc->ijkabc",T3,W[o,v,o,v],optimize="optimal")
    rooovvv += 0.041666667 * np.einsum("ijkade,debc->ijkabc",T3,W[v,v,v,v],optimize="optimal")


    nocc=nvir=None
    rooovvv=tamps.antisym_T3(rooovvv,nocc,nvir)
    return rooovvv


