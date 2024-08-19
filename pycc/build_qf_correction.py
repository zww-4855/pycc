import numpy as np
import pycc.tamps as tamps
from numpy import einsum


def WnT2sqr_toT2(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)

    roovv = -0.500000000 * np.einsum("ikab,jlcd,celm,mdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,jlcd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,lmcd,celm,jdke->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikab,lmcd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klab,ijcd,cekm,mdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klab,imcd,cekm,jdle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klab,imcd,cdkn,jnlm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,mncd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,klbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ijac,klbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,klde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,jlbd,efkl,cdef->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,jlbd,dekm,mcle->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,jlbd,cdmn,mnkl->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,cdln,jnkm->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlde,dekm,mclb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlde,cdlm,mekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,lmde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,lmde,cdlm,jekb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,imbd,dekl,jcme->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,mnbd,cdkm,ijln->ijab",T2,T2,T2dag,W[o,o,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,ijde,dfkl,cebf->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,ijde,cdkm,melb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,imde,dekl,jcmb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += 1.000000000 * np.einsum("klac,imde,cdkm,jelb->ijab",T2,T2,T2dag,W[o,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,klef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,jlef,cekl,dfab->ijab",T2,T2,T2dag,W[v,v,v,v],optimize="optimal")


    roovv = tamps.antisym_T2(roovv,None,None)
    return roovv



def WnT3_toT2(W,T3,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)

    roovv = 0.125000000 * np.einsum("cdkl,ijmabd,klmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ilmabd,jkmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,klmabd,ijmc->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlabe,kecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklabe,jecd->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,ijmacd,klmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,ilmacd,jkmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,klmacd,ijmb->ijab",T2dag,T3,W[o,o,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("cdkl,ijlade,kebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("cdkl,iklade,jebc->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("cdkl,ijlcde,keab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("cdkl,iklcde,jeab->ijab",T2dag,T3,W[o,v,v,v],optimize="optimal")

    roovv = tamps.antisym_T2(roovv,None,None)
    return roovv


def wnT2T3_toT4(W,T2,T3,o,v):
    roooovvvv = -0.125000000 * np.einsum("imab,jkncde,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("imab,jklcef,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("mnab,ijkcde,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.041666667 * np.einsum("ijae,kmnbcd,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += -0.125000000 * np.einsum("ijae,klmbcf,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("imae,jknbcd,lemn->ijklabcd",T2,T3,W[o,v,o,o],optimize="optimal")
    roooovvvv += 0.083333333 * np.einsum("imae,jklbcf,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")
    roooovvvv += -0.020833333 * np.einsum("ijef,klmabc,efmd->ijklabcd",T2,T3,W[v,v,o,v],optimize="optimal")

    roooovvvv=tamps.antisym_T4(roooovvvv,None,None)
    return roooovvvv


def wnT2cubed_toT4(W,T2,o,v):
    no=np.shape(T2)[0]
    nv=np.shape(T2)[2]
    roooovvvv = np.zeros((no,no,no,no,nv,nv,nv,nv))
    roooovvvv += -0.031250000 * np.einsum("imab,jncd,klef,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += 0.250000000 * np.einsum("imab,jkce,lndf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")
    roooovvvv += -0.031250000 * np.einsum("mnab,ijce,kldf,efmn->ijklabcd",T2,T2,T2,W[v,v,o,o],optimize="optimal")

    roooovvvv=tamps.antisym_T4(roooovvvv,None,None)
    return roooovvvv


def wnT2cubed_toT2(W,T2,o,v):
    T2dag=T2.transpose(2,3,0,1)
    no=np.shape(T2)[0]
    nv=np.shape(T2)[2]
    roovv = np.zeros((no,no,nv,nv))
    roovv += 0.500000000 * np.einsum("ikab,jlcd,mnef,celm,dfkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,jlcd,mnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikab,jlcd,mnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ikab,lmcd,jnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klab,ijcd,mnef,cekm,dfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,ijcd,mnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,ijcd,mnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,imcd,jnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klab,imcd,jnef,cekn,dflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,imcd,jnef,cemn,dfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ijac,klbd,mnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ijac,klbd,mnef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,klbd,mnef,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,klbd,mnef,demn,cfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ikac,jlbd,mnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jlbd,mnef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,mnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,mnef,demn,cfkl->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ikac,lmbd,jnef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,lmbd,jnef,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,lmbd,jnef,dekn,cflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,lmbd,jnef,eflm,cdkn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,lmbd,jnef,deln,cfkm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,imbd,jnef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("klac,imbd,jnef,dekn,cflm->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klac,mnbd,ijef,dekl,cfmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,mnbd,ijef,dekm,cfln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,klae,mnbf,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,jlae,mnbf,efkm,cdln->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,imae,jnbf,efkl,cdmn->ijab",T2,T2,T2,T2dag,W[v,v,o,o],optimize="optimal")


    roovv=tamps.antisym_T2(roovv,None,None)
    return roovv


def wnT2T3_toT2(W,T2,T3,o,v):
    T2dag=T2.transpose(2,3,0,1)
    roovv = 0.250000000 * np.einsum("ikab,cdlm,jmncde,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,cdlm,lmncde,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikab,cdlm,jlmdef,efkc->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,cdkm,ijncde,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klab,cdkm,imncde,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klab,cdkm,ijmdef,eflc->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,cdmn,ijncde,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klab,cdmn,imncde,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,dekl,lmnbde,kcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijac,dekl,klmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijac,dekl,klmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikac,dekl,jmnbde,lcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,dekl,lmnbde,jcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,dekl,jlmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,dekl,jlmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,cdlm,jmnbde,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,cdlm,lmnbde,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikac,cdlm,jlmbef,efkd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,cdlm,jlmdef,efkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,delm,jmnbde,lckn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,delm,lmnbde,jckn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikac,delm,jlmbef,cfkd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikac,delm,jlmdef,cfkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klac,dekl,imnbde,jcmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekl,ijmbef,cfmd->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klac,dekl,ijmdef,cfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("klac,cdkm,ijnbde,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("klac,cdkm,imnbde,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,cdkm,ijmbef,efld->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,cdkm,ijmdef,eflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekm,ijnbde,mcln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,dekm,imnbde,jcln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klac,dekm,ijmbef,cfld->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,dekm,ijmdef,cflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("klac,cdmn,ijnbde,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klac,cdmn,imnbde,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,cekl,lmnabe,kdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ijcd,cekl,klmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijcd,cekl,klmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("ijcd,efkl,klmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.062500000 * np.einsum("ijcd,efkl,klmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikcd,cekl,jmnabe,ldmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,cekl,lmnabe,jdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikcd,cekl,jlmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikcd,cekl,jlmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikcd,efkl,jlmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,efkl,jlmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,cdlm,jmnabe,lekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("ikcd,cdlm,lmnabe,jekn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ikcd,cdlm,jlmaef,efkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,celm,jmnabe,ldkn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,celm,lmnabe,jdkn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ikcd,celm,jlmabf,dfke->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ikcd,celm,jlmaef,dfkb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,cekl,imnabe,jdmn->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klcd,cekl,ijmabf,dfme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekl,ijmaef,dfmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klcd,efkl,ijmabf,cdme->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klcd,efkl,ijmaef,cdmb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.125000000 * np.einsum("klcd,cdkm,ijnabe,meln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cdkm,imnabe,jeln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klcd,cdkm,ijmaef,eflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekm,ijnabe,mdln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klcd,cekm,imnabe,jdln->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("klcd,cekm,ijmabf,dfle->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("klcd,cekm,ijmaef,dflb->ijab",T2,T2dag,T3,W[v,v,o,v],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klcd,cdmn,ijnabe,mekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")
    roovv += -0.062500000 * np.einsum("klcd,cdmn,imnabe,jekl->ijab",T2,T2dag,T3,W[o,v,o,o],optimize="optimal")

    roovv=tamps.antisym_T2(roovv,None,None)
    return roovv



def wnT2T3_pdagq(g,l2,t2,t3,o,v):
    #	  0.2500 P(i,j)<n,m||e,l>*l2(k,l,d,c)*t2(a,b,i,n)*t3(e,d,c,j,k,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmel,kldc,abin,edcjkm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,l>*l2(k,l,d,c)*t2(a,b,k,n)*t3(e,d,c,i,j,m)
    doubles_res +=  0.250000000000000 * einsum('nmel,kldc,abkn,edcijm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,l>*l2(k,l,d,c)*t2(a,b,n,m)*t3(e,d,c,i,j,k)
    doubles_res +=  0.125000000000000 * einsum('nmel,kldc,abnm,edcijk->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(c,b,i,n)*t3(e,d,a,j,k,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmel,kldc,cbin,edajkm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(c,b,k,n)*t3(e,d,a,i,j,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmel,kldc,cbkn,edaijm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(c,b,n,m)*t3(e,d,a,i,j,k)
    contracted_intermediate = -0.250000000000000 * einsum('nmel,kldc,cbnm,edaijk->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(i,j)<n,m||e,l>*l2(k,l,d,c)*t2(d,c,i,n)*t3(e,a,b,j,k,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmel,kldc,dcin,eabjkm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <n,m||e,l>*l2(k,l,d,c)*t2(d,c,k,n)*t3(e,a,b,i,j,m)
    doubles_res +=  0.250000000000000 * einsum('nmel,kldc,dckn,eabijm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <n,m||e,l>*l2(k,l,d,c)*t2(d,c,n,m)*t3(e,a,b,i,j,k)
    doubles_res +=  0.125000000000000 * einsum('nmel,kldc,dcnm,eabijk->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(e,b,i,j)*t3(d,c,a,k,n,m)
    contracted_intermediate = -0.125000000000000 * einsum('nmel,kldc,ebij,dcaknm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.1250 P(i,j)*P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(e,b,i,k)*t3(d,c,a,j,n,m)
    contracted_intermediate =  0.125000000000000 * einsum('nmel,kldc,ebik,dcajnm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(e,b,i,n)*t3(d,c,a,j,k,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmel,kldc,ebin,dcajkm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<n,m||e,l>*l2(k,l,d,c)*t2(e,b,k,n)*t3(d,c,a,i,j,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmel,kldc,ebkn,dcaijm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 <n,m||e,l>*l2(k,l,d,c)*t2(e,c,i,j)*t3(d,a,b,k,n,m)
    doubles_res += -0.250000000000000 * einsum('nmel,kldc,ecij,dabknm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<n,m||e,l>*l2(k,l,d,c)*t2(e,c,i,k)*t3(d,a,b,j,n,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmel,kldc,ecik,dabjnm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,l>*l2(k,l,d,c)*t2(e,c,i,n)*t3(d,a,b,j,k,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmel,kldc,ecin,dabjkm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,l>*l2(k,l,d,c)*t2(e,c,k,n)*t3(d,a,b,i,j,m)
    doubles_res += -0.500000000000010 * einsum('nmel,kldc,eckn,dabijm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(a,b,i,n)*t3(e,d,c,k,l,m)
    contracted_intermediate =  0.125000000000000 * einsum('nmej,kldc,abin,edcklm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(a,b,k,n)*t3(e,d,c,i,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmej,kldc,abkn,edcilm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.0625 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(a,b,n,m)*t3(e,d,c,i,k,l)
    contracted_intermediate =  0.062500000000000 * einsum('nmej,kldc,abnm,edcikl->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(c,b,i,n)*t3(e,d,a,k,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmej,kldc,cbin,edaklm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(c,b,k,n)*t3(e,d,a,i,l,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,kldc,cbkn,edailm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(c,b,n,m)*t3(e,d,a,i,k,l)
    contracted_intermediate = -0.125000000000000 * einsum('nmej,kldc,cbnm,edaikl->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.1250 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(d,c,i,n)*t3(e,a,b,k,l,m)
    contracted_intermediate =  0.125000000000000 * einsum('nmej,kldc,dcin,eabklm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(d,c,k,n)*t3(e,a,b,i,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmej,kldc,dckn,eabilm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.0625 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(d,c,n,m)*t3(e,a,b,i,k,l)
    contracted_intermediate =  0.062500000000000 * einsum('nmej,kldc,dcnm,eabikl->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(e,b,i,k)*t3(d,c,a,l,n,m)
    contracted_intermediate = -0.125000000000000 * einsum('nmej,kldc,ebik,dcalnm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.0625 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(e,b,k,l)*t3(d,c,a,i,n,m)
    contracted_intermediate = -0.062500000000000 * einsum('nmej,kldc,ebkl,dcainm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(e,b,i,n)*t3(d,c,a,k,l,m)
    contracted_intermediate = -0.125000000000000 * einsum('nmej,kldc,ebin,dcaklm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<n,m||e,j>*l2(k,l,d,c)*t2(e,b,k,n)*t3(d,c,a,i,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('nmej,kldc,ebkn,dcailm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(e,c,i,k)*t3(d,a,b,l,n,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmej,kldc,ecik,dablnm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(e,c,k,l)*t3(d,a,b,i,n,m)
    contracted_intermediate = -0.125000000000000 * einsum('nmej,kldc,eckl,dabinm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(e,c,i,n)*t3(d,a,b,k,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('nmej,kldc,ecin,dabklm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,j>*l2(k,l,d,c)*t2(e,c,k,n)*t3(d,a,b,i,l,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,kldc,eckn,dabilm->abij', g[o, o, v, o], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)<m,d||e,f>*l2(k,l,d,c)*t2(a,b,i,m)*t3(e,f,c,j,k,l)
    contracted_intermediate = -0.125000000000000 * einsum('mdef,kldc,abim,efcjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.2500 <m,d||e,f>*l2(k,l,d,c)*t2(a,b,k,m)*t3(e,f,c,i,j,l)
    doubles_res += -0.250000000000000 * einsum('mdef,kldc,abkm,efcijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.1250 P(i,j)*P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(c,b,i,m)*t3(e,f,a,j,k,l)
    contracted_intermediate =  0.125000000000000 * einsum('mdef,kldc,cbim,efajkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(c,b,k,m)*t3(e,f,a,i,j,l)
    contracted_intermediate =  0.250000000000000 * einsum('mdef,kldc,cbkm,efaijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(f,b,i,j)*t3(e,c,a,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('mdef,kldc,fbij,ecaklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(f,b,i,k)*t3(e,c,a,j,l,m)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,kldc,fbik,ecajlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(f,b,k,l)*t3(e,c,a,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('mdef,kldc,fbkl,ecaijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(f,b,i,m)*t3(e,c,a,j,k,l)
    contracted_intermediate = -0.250000000000000 * einsum('mdef,kldc,fbim,ecajkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,d||e,f>*l2(k,l,d,c)*t2(f,b,k,m)*t3(e,c,a,i,j,l)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,kldc,fbkm,ecaijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 <m,d||e,f>*l2(k,l,d,c)*t2(f,c,i,j)*t3(e,a,b,k,l,m)
    doubles_res +=  0.250000000000000 * einsum('mdef,kldc,fcij,eabklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)<m,d||e,f>*l2(k,l,d,c)*t2(f,c,i,k)*t3(e,a,b,j,l,m)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,kldc,fcik,eabjlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <m,d||e,f>*l2(k,l,d,c)*t2(f,c,k,l)*t3(e,a,b,i,j,m)
    doubles_res +=  0.250000000000000 * einsum('mdef,kldc,fckl,eabijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	 -0.2500 P(i,j)<m,d||e,f>*l2(k,l,d,c)*t2(f,c,i,m)*t3(e,a,b,j,k,l)
    contracted_intermediate = -0.250000000000000 * einsum('mdef,kldc,fcim,eabjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <m,d||e,f>*l2(k,l,d,c)*t2(f,c,k,m)*t3(e,a,b,i,j,l)
    doubles_res += -0.500000000000010 * einsum('mdef,kldc,fckm,eabijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.1250 <m,d||e,f>*l2(k,l,d,c)*t2(e,f,i,j)*t3(c,a,b,k,l,m)
    doubles_res +=  0.125000000000000 * einsum('mdef,kldc,efij,cabklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.2500 P(i,j)<m,d||e,f>*l2(k,l,d,c)*t2(e,f,i,k)*t3(c,a,b,j,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('mdef,kldc,efik,cabjlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.1250 <m,d||e,f>*l2(k,l,d,c)*t2(e,f,k,l)*t3(c,a,b,i,j,m)
    doubles_res +=  0.125000000000000 * einsum('mdef,kldc,efkl,cabijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1250 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(c,b,i,m)*t3(e,f,d,j,k,l)
    contracted_intermediate = -0.125000000000000 * einsum('maef,kldc,cbim,efdjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(c,b,k,m)*t3(e,f,d,i,j,l)
    contracted_intermediate = -0.250000000000000 * einsum('maef,kldc,cbkm,efdijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.0625 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(d,c,i,m)*t3(e,f,b,j,k,l)
    contracted_intermediate = -0.062500000000000 * einsum('maef,kldc,dcim,efbjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.1250 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(d,c,k,m)*t3(e,f,b,i,j,l)
    contracted_intermediate = -0.125000000000000 * einsum('maef,kldc,dckm,efbijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.1250 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,b,i,j)*t3(e,d,c,k,l,m)
    contracted_intermediate =  0.125000000000000 * einsum('maef,kldc,fbij,edcklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,b,i,k)*t3(e,d,c,j,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('maef,kldc,fbik,edcjlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.1250 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,b,k,l)*t3(e,d,c,i,j,m)
    contracted_intermediate =  0.125000000000000 * einsum('maef,kldc,fbkl,edcijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,b,i,m)*t3(e,d,c,j,k,l)
    contracted_intermediate = -0.125000000000000 * einsum('maef,kldc,fbim,edcjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,b,k,m)*t3(e,d,c,i,j,l)
    contracted_intermediate = -0.250000000000000 * einsum('maef,kldc,fbkm,edcijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,c,i,j)*t3(e,d,b,k,l,m)
    contracted_intermediate = -0.250000000000000 * einsum('maef,kldc,fcij,edbklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,c,i,k)*t3(e,d,b,j,l,m)
    contracted_intermediate =  0.500000000000010 * einsum('maef,kldc,fcik,edbjlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.2500 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,c,k,l)*t3(e,d,b,i,j,m)
    contracted_intermediate = -0.250000000000000 * einsum('maef,kldc,fckl,edbijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.2500 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,c,i,m)*t3(e,d,b,j,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('maef,kldc,fcim,edbjkl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(f,c,k,m)*t3(e,d,b,i,j,l)
    contracted_intermediate =  0.500000000000010 * einsum('maef,kldc,fckm,edbijl->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.0625 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(e,f,i,j)*t3(d,c,b,k,l,m)
    contracted_intermediate =  0.062500000000000 * einsum('maef,kldc,efij,dcbklm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.1250 P(i,j)*P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(e,f,i,k)*t3(d,c,b,j,l,m)
    contracted_intermediate = -0.125000000000000 * einsum('maef,kldc,efik,dcbjlm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.0625 P(a,b)<m,a||e,f>*l2(k,l,d,c)*t2(e,f,k,l)*t3(d,c,b,i,j,m)
    contracted_intermediate =  0.062500000000000 * einsum('maef,kldc,efkl,dcbijm->abij', g[o, v, v, v], l2, t2, t3, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    
    return doubles_res


def wnT3_pdagq(g,l2,t3,o,v):
    #	  0.5000 <m,d||k,l>*l2(k,l,d,c)*t3(c,a,b,i,j,m)
    doubles_res =  0.500000000000000 * einsum('mdkl,kldc,cabijm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)<m,d||j,l>*l2(k,l,d,c)*t3(c,a,b,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdjl,kldc,cabikm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <m,d||i,j>*l2(k,l,d,c)*t3(c,a,b,k,l,m)
    doubles_res +=  0.500000000000000 * einsum('mdij,kldc,cabklm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 P(a,b)<m,a||k,l>*l2(k,l,d,c)*t3(d,c,b,i,j,m)
    contracted_intermediate =  0.250000000000000 * einsum('makl,kldc,dcbijm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,a||j,l>*l2(k,l,d,c)*t3(d,c,b,i,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('majl,kldc,dcbikm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<m,a||i,j>*l2(k,l,d,c)*t3(d,c,b,k,l,m)
    contracted_intermediate =  0.250000000000000 * einsum('maij,kldc,dcbklm->abij', g[o, v, o, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.5000 <d,c||e,l>*l2(k,l,d,c)*t3(e,a,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('dcel,kldc,eabijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 P(i,j)<d,c||e,j>*l2(k,l,d,c)*t3(e,a,b,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('dcej,kldc,eabikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<d,a||e,l>*l2(k,l,d,c)*t3(e,c,b,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('dael,kldc,ecbijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<d,a||e,j>*l2(k,l,d,c)*t3(e,c,b,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('daej,kldc,ecbikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||e,l>*l2(k,l,d,c)*t3(e,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('abel,kldc,edcijk->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 P(i,j)<a,b||e,j>*l2(k,l,d,c)*t3(e,d,c,i,k,l)
    contracted_intermediate =  0.250000000000000 * einsum('abej,kldc,edcikl->abij', g[v, v, v, o], l2, t3, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    return doubles_res
