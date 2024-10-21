import numpy as np
import pycc.tamps as tamps

def T1resid_eqn(F,W,T1,T2,o,v):
    T1dag=T1.transpose(1,0)
    T2dag=T2.transpose(2,3,0,1)

    # Q1 [T1^,W]
    rov = 1.000000000 * np.einsum("bj,ijab->ia",T1dag,W[o,o,v,v],optimize="optimal")

    #Q1 [T1^,[W,T1]]
    rov1 = -1.000000000 * np.einsum("ja,bk,ikjb->ia",T1,T1dag,W[o,o,o,v],optimize="optimal")
    rov1 += -1.000000000 * np.einsum("ib,cj,jbac->ia",T1,T1dag,W[o,v,v,v],optimize="optimal")
    rov1 += 1.000000000 * np.einsum("jb,cj,ibac->ia",T1,T1dag,W[o,v,v,v],optimize="optimal")
    rov1 += 1.000000000 * np.einsum("jb,bk,ikja->ia",T1,T1dag,W[o,o,o,v],optimize="optimal")
    rov += rov1

    #Q1 0.5*[[W,T1],T1]
    rov2 = -2.000000000 * np.einsum("ja,kb,ibjk->ia",T1,T1,W[o,v,o,o],optimize="optimal")
    rov2 += -2.000000000 * np.einsum("ib,jc,bcja->ia",T1,T1,W[v,v,o,v],optimize="optimal")
    rov += rov2*0.5

    # Q1 [T1^,[W,T2]]
    rov3 = 0.500000000 * np.einsum("bj,klab,ijkl->ia",T1dag,T2,W[o,o,o,o],optimize="optimal")
    rov3 += -1.000000000 * np.einsum("bj,ikac,jckb->ia",T1dag,T2,W[o,v,o,v],optimize="optimal")
    rov3 += 1.000000000 * np.einsum("bj,jkac,ickb->ia",T1dag,T2,W[o,v,o,v],optimize="optimal")
    rov3 += 1.000000000 * np.einsum("bj,ikbc,jcka->ia",T1dag,T2,W[o,v,o,v],optimize="optimal")
    rov3 += -1.000000000 * np.einsum("bj,jkbc,icka->ia",T1dag,T2,W[o,v,o,v],optimize="optimal")
    rov3 += 0.500000000 * np.einsum("bj,ijcd,cdab->ia",T1dag,T2,W[v,v,v,v],optimize="optimal")
    rov += rov3

    #Q1 [W,T2]
    rov4 = -0.500000000 * np.einsum("jkab,ibjk->ia",T2,W[o,v,o,o],optimize="optimal")
    rov4 += -0.500000000 * np.einsum("ijbc,bcja->ia",T2,W[v,v,o,v],optimize="optimal")
    rov += rov4

    #Q1 [H,T1]
    rov5 = -1.000000000 * np.einsum("ij,ja->ia",F[o,o],T1,optimize="optimal")
    rov5 += 1.000000000 * np.einsum("ba,ib->ia",F[v,v],T1,optimize="optimal")
    rov5 += -1.000000000 * np.einsum("jb,ibja->ia",T1,W[o,v,o,v],optimize="optimal")
    rov += rov5

    # Q1 0.5*[T1^,[[W,T1],T1]]
    #rov6 = 2.000000000 * np.einsum("ja,ib,ck,kbjc->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    rov6 = -2.000000000 * np.einsum("ja,kb,ck,ibjc->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    rov6 += 2.000000000 * np.einsum("ja,kb,bl,iljk->ia",T1,T1,T1dag,W[o,o,o,o],optimize="optimal")
    rov6 += 2.000000000 * np.einsum("ib,jc,dj,bcad->ia",T1,T1,T1dag,W[v,v,v,v],optimize="optimal")
    rov6 += -2.000000000 * np.einsum("ib,jc,ck,kbja->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    #rov6 += 2.000000000 * np.einsum("jb,kc,cj,ibka->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    rov += rov6*0.5

    # Q1 [T1^,[[W,T1],T1]] -- only leftover, (T1^W)T1^2 portion 
    rov7 = 2.000000000 * np.einsum("ja,ib,ck,kbjc->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    rov7 += 2.000000000 * np.einsum("jb,kc,cj,ibka->ia",T1,T1,T1dag,W[o,v,o,v],optimize="optimal")
    rov += rov7

    return rov


def T2resid_eqn(F,W,T1,T2,o,v):
    T1dag=T1.transpose(1,0)
    T2dag=T2.transpose(2,3,0,1)

    # Q2 W|0>
    roovv = 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")

    # Q2 0.5*[[W,T1],T1]
    roovv2 = 0.500000000 * np.einsum("ka,lb,ijkl->ijab",T1,T1,W[o,o,o,o],optimize="optimal")
    roovv2 += 2.000000000 * np.einsum("ka,ic,jckb->ijab",T1,T1,W[o,v,o,v],optimize="optimal")
    roovv2 += 0.500000000 * np.einsum("ic,jd,cdab->ijab",T1,T1,W[v,v,v,v],optimize="optimal")
    roovv = roovv + 0.5*roovv2

    # Q1 [W,T1]
    roovv3 = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv3 += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += roovv3

    # Q2 [W,T2]
    roovv4 = 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv4 += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv4 += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += roovv4

    # Q2 [F,T2]
    roovv5 = 0.500000000 * np.einsum("ik,jkab->ijab",F[o,o],T2,optimize="optimal")
    roovv5 += -0.500000000 * np.einsum("ca,ijbc->ijab",F[v,v],T2,optimize="optimal")
    roovv += roovv5

    #roovv=tamps.antisym_T2(roovv,None,None)
    return roovv

def subtractOff_T1sqr(W,T1,T2,o,v):
    T1dag=T1.transpose(1,0)
    T2dag=T2.transpose(2,3,0,1)

    # -0.5*[T1^,[T1^,W]]
    term1 = 1.000000000 * np.einsum("ai,bj,ijab->",T1dag,T1dag,W[o,o,v,v],optimize="optimal")
    term1 = -0.5*term1

    # -0.5*[T1^,[T1^,[W,T1]]]
    term2 = 2.000000000 * np.einsum("ia,bj,ci,jabc->",T1,T1dag,T1dag,W[o,v,v,v],optimize="optimal")
    term2 += 2.000000000 * np.einsum("ia,bj,ak,jkib->",T1,T1dag,T1dag,W[o,o,o,v],optimize="optimal")
    term2 = -0.5*term2

    # -0.5*[T1^,[T1^,[W,T2]]]
    term3 = 0.500000000 * np.einsum("ai,bj,klab,ijkl->",T1dag,T1dag,T2,W[o,o,o,o],optimize="optimal")
    term3 += 2.000000000 * np.einsum("ai,bj,jkac,ickb->",T1dag,T1dag,T2,W[o,v,o,v],optimize="optimal")
    #term3 += -2.000000000 * np.einsum("ai,bj,jkbc,icka->",T1dag,T1dag,T2,W[o,v,o,v],optimize="optimal")
    term3 += 0.500000000 * np.einsum("ai,bj,ijcd,cdab->",T1dag,T1dag,T2,W[v,v,v,v],optimize="optimal")
    term3 = -0.5*term3

    # -0.25*[T1^,[T1^,[[W,T1],T1]]]
    term4 = -2.000000000 * np.einsum("ia,jb,cj,di,abcd->",T1,T1,T1dag,T1dag,W[v,v,v,v],optimize="optimal")
    term4 += 2.000000000 * np.einsum("ia,jb,ak,bl,klij->",T1,T1,T1dag,T1dag,W[o,o,o,o],optimize="optimal")
    term4 += -4.000000000 * np.einsum("ia,jb,bk,ci,kajc->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    #term4 += 4.000000000 * np.einsum("ia,jb,ck,bi,kajc->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    term4 = -0.25*term4

    # -0.5*[T1^,[T1^,[[W,T1],T1]]] -- the other part where (T1^W)T1^T1^2
    term5 = 4.000000000 * np.einsum("ia,jb,ck,bi,kajc->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    term5 = -0.5*term5

    print('subtracted off total:', term1+term2+term3+term4+term5)
    return term1+term2+term3+term4+term5


def test_3A(F,W,T1,T2,o,v):
    T1dag = T1.transpose(1,0)
    T2dag = T2.transpose(2,3,0,1)

    r = -2.000000000 * np.einsum("ia,jb,aj,bi->",F[o,v],T1,T1dag,T1dag,optimize="optimal")
    print('r for 3a is:', 0.3333*r)
 
    newr = 2.000000000 * np.einsum("ai,ib,ja,bj->",F[v,o],T1,T1,T1dag,optimize="optimal")
    print('newr is: ', newr*(1.0/6.0))

    beg = -2.000000000 * np.einsum("ai,ib,ja,bj->",F[v,o],T1,T1,T1dag,optimize="optimal")
    print('beg is: ', 0.5*beg)



    ## next diagram
    term1 = 2.000000000 * np.einsum("ji,ka,ib,bk,aj->",F[o,o],T1,T1,T1dag,T1dag,optimize="optimal")
    term1 += -2.000000000 * np.einsum("ba,ic,jb,cj,ai->",F[v,v],T1,T1,T1dag,T1dag,optimize="optimal")
    term1=term1*(1.0/3.0)

    term2 = -2.000000000 * np.einsum("ji,ka,ib,bk,aj->",F[o,o],T1,T1,T1dag,T1dag,optimize="optimal")
    term2 += 2.000000000 * np.einsum("ba,ic,jb,cj,ai->",F[v,v],T1,T1,T1dag,T1dag,optimize="optimal")
    term2 = term2*(1.0/6.0)
    print('term 1 is: ', term1,term2,term1+term2)

    r = 4.000000000 * np.einsum("ia,jb,ck,bi,kajc->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    print('scalar mag:', r*0.25)

    r = 2.000000000 * np.einsum("ia,jb,bk,cj,kaic->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    print('next scalar mag:', r*0.33333)

    r = -2.000000000 * np.einsum("ia,jb,ck,bi,kajc->",T1,T1,T1dag,T1dag,W[o,v,o,v],optimize="optimal")
    print('next scalar mag:', r*0.66666)

def test_cancel(F,W,T1,T2,o,v):
    T1dag = T1.transpose(1,0)
    T2dag = T2.transpose(2,3,0,1)
    r = 1.000000000 * np.einsum("ji,ka,ib,abjk->",F[o,o],T1,T1,T2dag,optimize="optimal")
    r += -1.000000000 * np.einsum("ba,ic,jb,acij->",F[v,v],T1,T1,T2dag,optimize="optimal")
    r = 0.5*r
    print('r', r)


    r2a = -1.000000000 * np.einsum("ji,ak,bj,ikab->",F[o,o],T1dag,T1dag,T2,optimize="optimal")
    r2a += 1.000000000 * np.einsum("ba,ci,aj,ijbc->",F[v,v],T1dag,T1dag,T2,optimize="optimal")
    r2a = r2a*0.5

    print('r2a',r2a)

    r1= 2.000000000 * np.einsum("ji,ak,bj,ikab->",F[o,o],T1dag,T1dag,T2,optimize="optimal")
    r1+= -2.000000000 * np.einsum("ba,ci,aj,ijbc->",F[v,v],T1dag,T1dag,T2,optimize="optimal")
    r1=r1*0.5
    print('r1:',r1)
    print('total:', r+r2a+r1)



    top = -1.000000000 * np.einsum("ai,bj,ijab->",F[v,o],T1dag,T2,optimize="optimal")
    top += 1.000000000 * np.einsum("ia,jb,abij->",F[o,v],T1,T2dag,optimize="optimal")
    top = top*0.5

    bottom = 1.000000000 * np.einsum("ai,bj,ijab->",F[v,o],T1dag,T2,optimize="optimal")
    print("top, bottom, total:", top, bottom, top+bottom)

    fin = -2.000000000 * np.einsum("ai,bj,jkbc,icka->",T1dag,T1dag,T2,W[o,v,o,v],optimize="optimal")
    print('final diagram:',fin*0.5)
