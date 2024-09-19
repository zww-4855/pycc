import numpy as np

# T2^(WT2^2)c

def t2dag_WnT2sqr_oooo(T2,Woooo):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("klab,mncd,cdkm,ijln->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("klac,mnbd,cdkm,ijln->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    return roovv

def t2dag_WnT2sqr_vvvv(T2,Wvvvv):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("ijcd,klef,cekl,dfab->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikcd,jlef,cekl,dfab->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    return roovv

def t2dag_WnT2sqr_ovov(T2,Wovov):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -1.000000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 2.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikac,lmde,dekl,jcmb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikac,lmde,cdlm,jekb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klac,imde,dekl,jcmb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 2.000000000 * np.einsum("klac,imde,cdkm,jelb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    return roovv

####################################################
###################################################
# t2dagwnC_T2sqr_oooo
def t2dagWnC_T2sqr_oooo(T2,Woooo):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("ikab,lmcd,cdkn,jnlm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("klab,imcd,cdmn,jnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("klac,imbd,cdmn,jnkl->ijab",T2,T2,T2dag,Woooo,optimize=("optimal",10**6))
    return roovv

def t2dagWnC_T2sqr_vvvv(T2,Wvvvv):
    T2dag=T2.transpose(2,3,0,1)
    roovv = -0.250000000 * np.einsum("ikab,jlcd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.062500000 * np.einsum("klab,ijcd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 0.500000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("ijcd,klae,efkl,cdbf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("ikcd,jlae,efkl,cdbf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    roovv += -0.250000000 * np.einsum("klcd,ijae,efkl,cdbf->ijab",T2,T2,T2dag,Wvvvv,optimize=("optimal",10**6))
    return roovv

def t2dagWnC_T2sqr_ovov(T2,Wovov):
    T2dag=T2.transpose(2,3,0,1)
    roovv = 1.000000000 * np.einsum("ikab,jlcd,cekm,mdle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikab,lmcd,cekl,jdme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klab,imcd,cekl,jdme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ijac,klde,cdkm,melb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 2.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,jlbd,delm,mcke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikac,lmbd,delm,jcke->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 2.000000000 * np.einsum("ikac,jlde,cdkm,melb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 1.000000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -2.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("ikac,jlde,delm,mckb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("klac,ijbd,dekm,mcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += 2.000000000 * np.einsum("klac,imbd,dekm,jcle->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -1.000000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    roovv += -0.500000000 * np.einsum("klac,ijde,dekm,mclb->ijab",T2,T2,T2dag,Wovov,optimize=("optimal",10**6))
    return roovv
