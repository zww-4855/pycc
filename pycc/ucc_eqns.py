import numpy as np
import pycc.tamps as tamps
import pycc.faster_ucc5eqns as faster_ucc5eqns
#from memory_profiler import memory_usage
#from memory_profiler import profile

def ucc_eqnDriver(calcType,Fock,W,T1,T2,o,v):
    if calcType == "UCCD3":
        D1T1 = T1 ## should be 0
        D2T2 = ucc3_t2resid(Fock,W,T2,o,v)

    if calcType == "UCCSD4" or calcType == "UCCD4":
        D1T1 = uccsd4_t1resid(Fock,W,T1,T2,o,v)
        D2T2 = uccsd4_t2resid(Fock,W,T1,T2,o,v)
        
    if calcType == "UCCSD5" or calcType == "UCCD5":
        D1T1 = uccsd4_t1resid(Fock,W,T1,T2,o,v)
        print('done w D1T1')
        print(flush=True)
        D1T1 += uccsd5_t1resid_linearTerms(W,T1,o,v)
        print('done w t1')
        print(flush=True)
        D1T1 += uccsd5_t1resid_quadTerm(W,T2,o,v)
        
        
        print('done w t1')
        print(flush=True)
        D2T2 =  uccsd4_t2resid(Fock,W,T1,T2,o,v)
        D2T2 += uccsd5_t2resid_T1couplings(W,T1,T2,o,v)
        print('done w t2')
        print(flush=True)
        D2T2 += 0.25*faster_ucc5eqns.t2dagWnC_T2sqr_ovov(T2,W[o,v,o,v])
        D2T2 += 0.25*faster_ucc5eqns.t2dag_WnT2sqr_ovov(T2,W[o,v,o,v])

        D2T2 += 0.25*faster_ucc5eqns.t2dagWnC_T2sqr_oooo(T2,W[o,o,o,o])
        D2T2 += 0.25*faster_ucc5eqns.t2dag_WnT2sqr_oooo(T2,W[o,o,o,o])

        D2T2 += 0.25*faster_ucc5eqns.t2dagWnC_T2sqr_vvvv(T2,W[v,v,v,v])
        D2T2 += 0.25*faster_ucc5eqns.t2dag_WnT2sqr_vvvv(T2,W[v,v,v,v])

        print('done w t2')
        print(flush=True)
#        mem_usage = memory_usage((uccsd5_t2resid_T1couplings,(W,T1,T2,o,v)))
#        print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
#        print('Maximum memory usage: %s' % max(mem_usage))


############################################################
        #D2T2 += 0.25*uccsd5_t2resid_t2dagwnC_t2sqr(W,T2,o,v) 
        #print('done w t2')
        #print(flush=True)
        #D2T2 += 0.25*uccsd5_t2resid_t2dag_wnt2sqrC(W,T2,o,v)
###########################################################

    nocc=nvir=None
    D2T2=tamps.antisym_T2(D2T2,nocc,nvir)
    return D1T1,D2T2

def ucc3_t2resid(Fock,W,T2,o,v):
    print('shape of T2:',np.shape(T2))
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

#@profile
def uccsd4_t2resid(Fock,W,T1,T2,o,v):
    # terms linear in T
    roovv = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += ucc3_t2resid(Fock,W,T2,o,v)

    # terms quadratic in T2
    tmp_wnT2sqr=0.5*uccsd_wnT2sqr(W,T2,o,v)
    roovv += tmp_wnT2sqr

    # T2dagWnT2, 0.5*Q2[[W,tau2],tau2], or 0.5*[[T2dag,W],T2]
    roovv += 0.5*uccsd_T2dagWnT2(W,T2,o,v)
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
    T1dag= T1.transpose(1,0)
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
    roovv += uccsd5_t1dagwnCt2(W,T1,T2,o,v)
    return roovv



def uccsd5_t1dagwnCt2(W,T1,T2,o,v):
    T1dag = T1.transpose(1,0)
    # T1^WnT2
    roovv = -0.500000000 * np.einsum("ck,ilab,jklc->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ck,klab,ijlc->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ck,ilac,jklb->ijab",T1dag,T2,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ck,ijad,kdbc->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ck,ikad,jdbc->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ck,ijcd,kdab->ijab",T1dag,T2,W[o,v,v,v],optimize="optimal")
    return roovv

#@profile
def uccsd5_t2resid_t2dagwnC_t2sqr(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)
    print('shape of T2:',np.shape(T2))
    roovv = -0.250000000 * np.einsum("ikab,jlcd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += 1.000000000 * np.einsum("ikab,jlcd,cekm,mdle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -1.000000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -1.000000000 * np.einsum("ikab,lmcd,cekl,jdme->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.250000000 * np.einsum("ikab,lmcd,cdkn,jnlm->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -0.500000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 0.500000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += 0.062500000 * np.einsum("klab,ijcd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += -0.500000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -0.500000000 * np.einsum("klab,imcd,cekl,jdme->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 1.000000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.500000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -0.250000000 * np.einsum("klab,imcd,cdmn,jnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -0.250000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += -1.000000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += 0.500000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += -1.000000000 * np.einsum("ijac,klde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.500000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 0.500000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += 2.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -2.000000000 * np.einsum("ikac,jlbd,delm,mcke->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -2.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -1.000000000 * np.einsum("ikac,lmbd,delm,jcke->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -2.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += -2.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += 2.000000000 * np.einsum("ikac,jlde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 1.000000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -2.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -1.000000000 * np.einsum("ikac,jlde,delm,mckb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -1.000000000 * np.einsum("klac,ijbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -1.000000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 2.000000000 * np.einsum("klac,imbd,dekm,jcle->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += 1.000000000 * np.einsum("klac,imbd,cdmn,jnkl->ijab",T2,T2,T2dag,W[o,o,o,o])
    roovv += 0.500000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += -1.000000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.500000000 * np.einsum("klac,ijde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v])
    roovv += -0.250000000 * np.einsum("ijcd,klae,efkl,cdbf->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += 1.000000000 * np.einsum("ikcd,jlae,efkl,cdbf->ijab",T2,T2,T2dag,W[v,v,v,v])
    roovv += -0.250000000 * np.einsum("klcd,ijae,efkl,cdbf->ijab",T2,T2,T2dag,W[v,v,v,v])
    return roovv

#@profile
def uccsd5_t2resid_t2dag_wnt2sqrC(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -1.000000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klab,mncd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 2.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -2.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -2.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -2.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -2.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmde,cdlm,jekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,mnbd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,imde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 2.000000000 * np.einsum("klac,imde,cdkm,jelb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijcd,klef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikcd,jlef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    return roovv
