import numpy as np

def wntau2comm_UCCSD4_t2resid(W,T2,o,v):
    T2dag = T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,cdkl,jlcd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.031250000 * np.einsum("klab,cdkl,ijcd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,cdkl,klbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,cdkl,jlbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klac,cdkl,ijbd->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += 0.031250000 * np.einsum("ijcd,cdkl,klab->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikcd,cdkl,jlab->ijab",T2,T2dag,W[o,o,v,v],optimize="optimal")
    return roovv


def focktau2comm_UCCSD4_t2resid(F,T2,o,v):
    T2dag = T2.transpose(2,3,0,1)
    roovv = -0.041666667 * np.einsum("ik,jlab,kmcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.041666667 * np.einsum("ik,klab,jmcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.020833333 * np.einsum("ik,lmab,jkcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.166666667 * np.einsum("ik,jlac,kmbd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += -0.083333333 * np.einsum("ik,lmac,jkbd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.125000000 * np.einsum("lk,ikab,jmcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += -0.125000000 * np.einsum("lk,imab,jkcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += -0.062500000 * np.einsum("lk,kmab,ijcd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.250000000 * np.einsum("lk,ijac,kmbd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.500000000 * np.einsum("lk,imac,jkbd,cdlm->ijab",F[o,o],T2,T2,T2dag,optimize="optimal")
    roovv += 0.041666667 * np.einsum("ca,ijbd,klce,dekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += -0.166666667 * np.einsum("ca,ikbd,jlce,dekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += 0.041666667 * np.einsum("ca,klbd,ijce,dekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += -0.020833333 * np.einsum("ca,ijde,klbc,dekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += 0.083333333 * np.einsum("ca,ikde,jlbc,dekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += -0.250000000 * np.einsum("dc,ikab,jlde,cekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += 0.062500000 * np.einsum("dc,klab,ijde,cekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += 0.125000000 * np.einsum("dc,ijae,klbd,cekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += -0.500000000 * np.einsum("dc,ikae,jlbd,cekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    roovv += 0.125000000 * np.einsum("dc,klae,ijbd,cekl->ijab",F[v,v],T2,T2,T2dag,optimize="optimal")
    return roovv


def wnt2commE_portionUCCSD4(W,T2,o,v):
    T2dag = T2.transpose(2,3,0,1)
    r = 0.041666667 * np.einsum("ijab,klcd,cdij,abkl->",T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    r += 0.333333333 * np.einsum("ijab,klcd,acik,bdjl->",T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    r += -0.166666667 * np.einsum("ijab,klcd,cdik,abjl->",T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    r += -0.166666667 * np.einsum("ijab,klcd,ackl,bdij->",T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    r += 0.333333333 * np.einsum("ijab,acik,bdjl,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.166666667 * np.einsum("ijab,cdjk,abil,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.041666667 * np.einsum("ijab,abkl,cdij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")
    r += 0.166666667 * np.einsum("ijab,bckl,adij,klcd->",T2,T2dag,T2dag,W[o,o,v,v],optimize="optimal")


    r += 0.125000000 * np.einsum("ijab,cdij,abcd->",T2,T2dag,W[v,v,v,v],optimize="optimal")
    r += -1.000000000 * np.einsum("ijab,acik,kbjc->",T2,T2dag,W[o,v,o,v],optimize="optimal")
    r += 0.125000000 * np.einsum("ijab,abkl,klij->",T2,T2dag,W[o,o,o,o],optimize="optimal")
    r += 0.250000000 * np.einsum("abij,ijab->",T2dag,W[o,o,v,v],optimize="optimal")


    #	  0.50 <k,i||j,i>*t2(b,a,j,l)*t2(a,b,l,k)
    energy =  0.50 * np.einsum('kiji,bajl,ablk', W[o, o, o, o], T2dag, T2dag, optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.50 <j,l||i,l>*t2(b,a,i,k)*t2(a,b,k,j)
    #energy += -0.50 * np.einsum('jlil,baik,abkj', W[o, o, o, o], T2dag, T2dag, optimize=['einsum_path', (1, 2), (0, 1)])

    print('SCF self look t2 energy:', energy)


#	  0.010416666666666666 <j,i||a,b>*t2(a,b,j,i)*t2(d,c,k,l)*t2(c,d,l,k)
    energy =  0.010416666666666666 * np.einsum('jiab,abji,dckl,cdlk', W[o, o, v, v], T2dag, T2dag, T2dag, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.010416666666666666 <b,a||i,j>*t2(a,b,j,i)*t2(d,c,k,l)*t2(c,d,l,k)
    energy +=  0.010416666666666666 * np.einsum('baij,abji,dckl,cdlk', W[v, v, o, o], T2dag, T2dag, T2dag, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.010416666666666666 <l,k||c,d>*t2(b,a,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.010416666666666666 * np.einsum('lkcd,baij,abji,cdlk', W[o, o, v, v], T2dag, T2dag, T2dag, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.010416666666666666 <d,c||k,l>*t2(b,a,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.010416666666666666 * np.einsum('dckl,baij,abji,cdlk', W[v, v, o, o], T2dag, T2dag, T2dag, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    print('unlinked diagram E:',energy)

    return r

def fockE_portionUCCSD4(F,T2,o,v):
    T2dag = T2.transpose(2,3,0,1)
    r = 0.166666667 * np.einsum("ji,klab,imcd,bdkl,acjm->",F[o,o],T2,T2,T2dag,T2dag,optimize="optimal")
    r += -0.041666667 * np.einsum("ji,klab,imcd,cdkl,abjm->",F[o,o],T2,T2,T2dag,T2dag,optimize="optimal")
    r += -0.083333333 * np.einsum("ji,klab,imcd,ablm,cdjk->",F[o,o],T2,T2,T2dag,T2dag,optimize="optimal")
    r += 0.333333333 * np.einsum("ji,klab,imcd,bdlm,acjk->",F[o,o],T2,T2,T2dag,T2dag,optimize="optimal")
    r += -0.083333333 * np.einsum("ji,klab,imcd,cdlm,abjk->",F[o,o],T2,T2,T2dag,T2dag,optimize="optimal")
    r += 0.083333333 * np.einsum("ba,ijcd,klbe,deij,ackl->",F[v,v],T2,T2,T2dag,T2dag,optimize="optimal")
    r += -0.166666667 * np.einsum("ba,ijcd,klbe,cdjl,aeik->",F[v,v],T2,T2,T2dag,T2dag,optimize="optimal")
    r += -0.333333333 * np.einsum("ba,ijcd,klbe,dejl,acik->",F[v,v],T2,T2,T2dag,T2dag,optimize="optimal")
    r += 0.041666667 * np.einsum("ba,ijcd,klbe,cdkl,aeij->",F[v,v],T2,T2,T2dag,T2dag,optimize="optimal")
    r += 0.083333333 * np.einsum("ba,ijcd,klbe,dekl,acij->",F[v,v],T2,T2,T2dag,T2dag,optimize="optimal")

    r += -0.500000000 * np.einsum("ji,ikab,abjk->",F[o,o],T2,T2dag,optimize="optimal")
    r += 0.500000000 * np.einsum("ba,ijbc,acij->",F[v,v],T2,T2dag,optimize="optimal")

    return r

def t1coupledE_portionUCCSD4(F,W,T1,T2,o,v):
    T1dag = T1.transpose(1,0)
    T2dag = T2.transpose(2,3,0,1)

    r = -1.000000000 * np.einsum("ji,ia,aj->",F[o,o],T1,T1dag,optimize="optimal")
    r += 1.000000000 * np.einsum("ba,ib,ai->",F[v,v],T1,T1dag,optimize="optimal")
    r += -1.000000000 * np.einsum("ia,bcij,jabc->",T1,T2dag,W[o,v,v,v],optimize="optimal")
    r += -1.000000000 * np.einsum("ia,abjk,jkib->",T1,T2dag,W[o,o,o,v],optimize="optimal")
    r += -1.000000000 * np.einsum("ai,jkab,ibjk->",T1dag,T2,W[o,v,o,o],optimize="optimal")
    r += -1.000000000 * np.einsum("ai,ijbc,bcja->",T1dag,T2,W[v,v,o,v],optimize="optimal")
    return 0.5*r
