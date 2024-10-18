import numpy as np
import pycc.tamps as tamps
import pycc.cc_energy as cc_energy
import pycc.misc as misc
from copy import deepcopy

def drive_pcc_energyCorrections(driveCCobj):#,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb):
    oa = driveCCobj.occSliceInfo["occ_aa"]
    ob = driveCCobj.occSliceInfo["occ_bb"]
    va = driveCCobj.occSliceInfo["virt_aa"]
    vb = driveCCobj.occSliceInfo["virt_bb"]


    W_aaaa = driveCCobj.integralInfo["tei_aaaa"]
    W_bbbb = driveCCobj.integralInfo["tei_bbbb"]
    W_abab = driveCCobj.integralInfo["tei_abab"]

    firstOrderwvfxn,E2 = firstOrder_Driver(driveCCobj,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)
    E3=secondOrder_Driver(driveCCobj,firstOrderwvfxn,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)


    print('\n\n\n\n\n ')
    print('**********************')
    print('Final E(2)+E(3) correction is: ', E2+E3)
    print("Total pUCC+E(2)+E(3) energy: ", driveCCobj.correlationE["totalE"]+E2+E3)

def firstOrder_Driver(driveCCobj,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    # Use this as our zeroth-order amplitudes, taken from a converged pCC calculation,
    # to build first order wavefxns w/ the appropriate set of integrals
    pcc_T2_ab = driveCCobj.tamps["t2ab"]

    # first build first-order correction to T2:
    first_orderT2(driveCCobj,W_aaaa,W_bbbb,W_abab,pcc_T2_ab,oa,ob,va,vb)
    T2_aa = driveCCobj.pcc_amps["FO_aa"]
    T2_bb = driveCCobj.pcc_amps["FO_bb"]
    T2_ab = driveCCobj.pcc_amps["FO_ab"]

    # Don't need to zero the off-diagonal of T2aaaa/bbbb, but will need to zero-out the diagonal of the newly constructed T2ab
    T2_ab = misc.zeroT2_offDiagonal(T2_ab)
    FO_energy_partA = cc_energy.spinIntegrated_CCDE(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb)
    print('First part of E(2) correction to pUCC:', FO_energy_partA)


    energy_aa = 0.250000000 * np.einsum("abij,ijab->",W_aaaa[va,va,oa,oa],T2_aa,optimize="optimal")
    energy_bb = 0.250000000 * np.einsum("ABIJ,IJAB->",W_bbbb[vb,vb,ob,ob],T2_bb,optimize="optimal")
    print('purely triplet-excited effects on E(2):',energy_aa,energy_bb)
    print('checking MP2-like diagram A:')
    mp2wvfxn_corrections, eA = FO_diagramA(driveCCobj,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)
    FOwvefxn_pUCC_V     , eB = FO_diagramB(driveCCobj,W_aaaa,W_bbbb,W_abab,driveCCobj.tamps["t2ab"],oa,ob,va,vb)
    eC                       = FO_diagramD(driveCCobj,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb) # need to check this function

    totalFO_E=eA+2.0*eB+eC
    print('E(2) diagram A:',eA)
    print('E(2) diagram B+C:',eB*2.0)
    print('E(2) diagram D:', eC)
    print('Total E(2):',totalFO_E)
    return {'mp2':mp2wvfxn_corrections, 'VpUCCt2':FOwvefxn_pUCC_V},totalFO_E


def first_orderT2(driveCCobj,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb):
    nocc = nvir = None
    # get all alpha, beta T2 corrections first
    # alpha first
    mp2_aa = 0.250000000 * np.einsum("ijab->ijab",W_aaaa[oa,oa,va,va],optimize="optimal")
    mp2_aa += 1.000000000 * np.einsum("iAaI,jIbA->ijab",W_abab[oa,vb,va,ob],T2_ab,optimize="optimal")

    # now get all beta corrections to T2
    mp2_bb = 0.250000000 * np.einsum("IJAB->IJAB",W_bbbb[ob,ob,vb,vb],optimize="optimal") 
    mp2_bb += 1.000000000 * np.einsum("aIiA,iJaB->IJAB",W_abab[va,ob,oa,vb],T2_ab,optimize="optimal")

    # now antisymmetrize all alpha,beta T2 first-order corrections 
    mp2_aa = tamps.antisym_T2(mp2_aa,nocc,nvir)
    mp2_bb = tamps.antisym_T2(mp2_bb,nocc,nvir)

    # aaaa/bbbb portion to FO T2, so save these
    driveCCobj.pcc_amps.update({"FO_aa":mp2_aa*driveCCobj.denomInfo["D2aabkup"]})
    driveCCobj.pcc_amps.update({"FO_bb":mp2_bb*driveCCobj.denomInfo["D2bbbkup"]})

    # now collect all ab terms, and save to dictionary
    FO_ab = -1.000000000 * np.einsum("ibja,jIbA->iIaA",W_aaaa[oa,va,oa,va],T2_ab,optimize="optimal")
    FO_ab += 1.000000000 * np.einsum("iIjJ,jJaA->iIaA",W_abab[oa,ob,oa,ob],T2_ab,optimize="optimal")
    FO_ab += -1.000000000 * np.einsum("iBjA,jIaB->iIaA",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    FO_ab += -1.000000000 * np.einsum("bIaJ,iJbA->iIaA",W_abab[va,ob,va,ob],T2_ab,optimize="optimal")
    FO_ab += 1.000000000 * np.einsum("iIaA->iIaA",W_abab[oa,ob,va,vb],optimize="optimal")
    FO_ab += 1.000000000 * np.einsum("bBaA,iIbB->iIaA",W_abab[va,vb,va,vb],T2_ab,optimize="optimal")
    FO_ab += -1.000000000 * np.einsum("IBJA,iJaB->iIaA",W_bbbb[ob,vb,ob,vb],T2_ab,optimize="optimal")


    driveCCobj.pcc_amps.update({"FO_ab":FO_ab*driveCCobj.denomInfo["D2abbkup"]})


    return 


def FO_diagramA(driveCCobj,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    nocc = nvir = None
    mp2_aa = 0.250000000 * np.einsum("ijab->ijab",W_aaaa[oa,oa,va,va],optimize="optimal")
    mp2_bb = 0.250000000 * np.einsum("IJAB->IJAB",W_bbbb[ob,ob,vb,vb],optimize="optimal")
    # now antisymmetrize all alpha,beta T2 first-order corrections
    mp2_aa = tamps.antisym_T2(mp2_aa,nocc,nvir)
    mp2_bb = tamps.antisym_T2(mp2_bb,nocc,nvir)

    FO_ab = 1.000000000 * np.einsum("iIaA->iIaA",W_abab[oa,ob,va,vb],optimize="optimal")

    # weigh these residuals against denoms
    mp2_aa = mp2_aa*driveCCobj.denomInfo["D2aabkup"]
    mp2_bb = mp2_bb*driveCCobj.denomInfo["D2bbbkup"]
    FO_ab =  FO_ab*driveCCobj.denomInfo["D2abbkup"]
    FO_ab = misc.zeroT2_offDiagonal(FO_ab)
    # now that we have T2 tensors, contract these against W
    FO_energy_partA = cc_energy.spinIntegrated_CCDE(W_aaaa,W_bbbb,W_abab,mp2_aa,mp2_bb,FO_ab,oa,ob,va,vb)
    print('First part of E(2) correction to pUCC:', FO_energy_partA)
    return {'t2aa':mp2_aa,'t2bb':mp2_bb,'t2ab':FO_ab}, FO_energy_partA

def FO_diagramB(driveCCobj,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb):
    nocc = nvir = None
    resid_aa = 1.000000000 * np.einsum("iAaI,jIbA->ijab",W_abab[oa,vb,va,ob],T2_ab,optimize="optimal")
    resid_bb = 1.000000000 * np.einsum("aIiA,iJaB->IJAB",W_abab[va,ob,oa,vb],T2_ab,optimize="optimal")
    # antisymmetrize these
    resid_aa = tamps.antisym_T2(resid_aa,nocc,nvir)
    resid_bb = tamps.antisym_T2(resid_bb,nocc,nvir)

    # Now get t2ab
    RoOvV = -1.000000000 * np.einsum("ibja,jIbA->iIaA",W_aaaa[oa,va,oa,va],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIjJ,jJaA->iIaA",W_abab[oa,ob,oa,ob],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iBjA,jIaB->iIaA",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bIaJ,iJbA->iIaA",W_abab[va,ob,va,ob],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBaA,iIbB->iIaA",W_abab[va,vb,va,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("IBJA,iJaB->iIaA",W_bbbb[ob,vb,ob,vb],T2_ab,optimize="optimal")

    # Now, contract all of them against D2, then zero-out RoOvV
    resid_aa=resid_aa*driveCCobj.denomInfo["D2aabkup"]
    resid_bb = resid_bb*driveCCobj.denomInfo["D2bbbkup"]
    RoOvV =  RoOvV*driveCCobj.denomInfo["D2abbkup"]
    RoOvV = misc.zeroT2_offDiagonal(RoOvV)

    # now that we have T2 tensors, contract these against W
    FO_energy_partA = cc_energy.spinIntegrated_CCDE(W_aaaa,W_bbbb,W_abab,resid_aa,resid_bb,RoOvV,oa,ob,va,vb)
    print('First part of E(2) correction to pUCC:', FO_energy_partA)

    FOwvefxn_pUCC_V={'t2aa':resid_aa,'t2bb':resid_bb,'t2ab':RoOvV}
    return FOwvefxn_pUCC_V,FO_energy_partA
    

def FO_diagramD(driveCCobj,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb):
    nocc = nvir = None
    resid_aa = 1.000000000 * np.einsum("iAaI,jIbA->ijab",W_abab[oa,vb,va,ob],T2_ab,optimize="optimal")
    resid_bb = 1.000000000 * np.einsum("aIiA,iJaB->IJAB",W_abab[va,ob,oa,vb],T2_ab,optimize="optimal")
    # antisymmetrize these
    resid_aa = tamps.antisym_T2(resid_aa,nocc,nvir)
    resid_bb = tamps.antisym_T2(resid_bb,nocc,nvir)


    # Now get t2ab
    RoOvV = -1.000000000 * np.einsum("ibja,jIbA->iIaA",W_aaaa[oa,va,oa,va],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIjJ,jJaA->iIaA",W_abab[oa,ob,oa,ob],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iBjA,jIaB->iIaA",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bIaJ,iJbA->iIaA",W_abab[va,ob,va,ob],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBaA,iIbB->iIaA",W_abab[va,vb,va,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("IBJA,iJaB->iIaA",W_bbbb[ob,vb,ob,vb],T2_ab,optimize="optimal")

    # Ensure the caps aren't overwritten
    cap_aa = deepcopy(resid_aa).transpose(2,3,0,1)
    cap_bb = deepcopy(resid_bb).transpose(2,3,0,1)
    cap_ab = deepcopy(RoOvV).transpose(2,3,0,1)

    # Now, contract all of the bases against D2, then zero-out RoOvV, cap_ab
    resid_aa=resid_aa*driveCCobj.denomInfo["D2aabkup"]
    resid_bb = resid_bb*driveCCobj.denomInfo["D2bbbkup"]
    RoOvV =  RoOvV*driveCCobj.denomInfo["D2abbkup"]
    RoOvV = misc.zeroT2_offDiagonal(RoOvV)

    # Determine energy contribution
    energy = 0.250000000 * np.einsum("abij,ijab->",cap_aa,resid_aa,optimize="optimal")
    energy += 1.000000000 * np.einsum("aAiI,iIaA->",cap_ab, RoOvV,optimize="optimal")
    energy += 0.250000000 * np.einsum("ABIJ,IJAB->",cap_bb,resid_bb,optimize="optimal")

    print('energy for FO D:', energy)
    return energy

def secondOrder_Driver(driveCCobj,firstOrderT2,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb):
    secondOrderWavefxn={}
    FOwvfxn_mp=firstOrderT2['mp2']
    FOwvfxn_VpUCC=firstOrderT2['VpUCCt2']
    FOwvfxn_aa = FOwvfxn_mp["t2aa"] + FOwvfxn_VpUCC["t2aa"]
    FOwvfxn_bb = FOwvfxn_mp["t2bb"] + FOwvfxn_VpUCC["t2bb"]
    FOwvfxn_ab = FOwvfxn_mp["t2ab"] + FOwvfxn_VpUCC["t2ab"]

    FOwvfxn_aa = tamps.antisym_T2(FOwvfxn_aa,None,None)
    FOwvfxn_bb = tamps.antisym_T2(FOwvfxn_bb,None,None)

    pucc_t2aa=driveCCobj.tamps["t2aa"]
    pucc_t2bb=driveCCobj.tamps["t2bb"]
    pucc_t2ab=driveCCobj.tamps["t2ab"]

    # First, build W(t2FO * pucct2)_D
    wvfxnA={}
    resid_aa = secondOrder_residaa(FOwvfxn_aa,FOwvfxn_bb,FOwvfxn_ab,pucc_t2aa,pucc_t2bb,pucc_t2ab,W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob],oa,ob,va,vb)
    resid_bb = secondOrder_residaa(FOwvfxn_aa,FOwvfxn_bb,FOwvfxn_ab,pucc_t2aa,pucc_t2bb,pucc_t2ab,W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob],oa,ob,va,vb)
    resid_ab = secondOrder_residab(FOwvfxn_aa,FOwvfxn_bb,FOwvfxn_ab,pucc_t2aa,pucc_t2bb,pucc_t2ab,W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob],oa,ob,va,vb)
    store_wavefxn_cap(driveCCobj,resid_aa,resid_bb,resid_ab,wvfxnA,prefactor=1.0) #     # Now build/store wavefunction and caps


    # Second, build pucct2^ (t2FO * W)_D
    wvfxnB={}
    T2aa_dag = pucc_t2aa.transpose(2,3,0,1)
    T2bbdag = pucc_t2bb.transpose(2,3,0,1)
    T2abdag = pucc_t2ab.transpose(2,3,0,1)
    T2aa     = FOwvfxn_aa 
    T2bb     = FOwvfxn_bb 
    T2ab     = FOwvfxn_ab 
    resid_aa = secondOrder_residaa(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    resid_bb = secondOrder_residbb(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    resid_ab = secondOrder_residab(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    store_wavefxn_cap(driveCCobj,resid_aa,resid_bb,resid_ab,wvfxnB,prefactor=0.5) #     # Now build/store wavefunction and caps

    # Lastly, build t2FO^ (puccct2 * W)_D
    wvfxnC={}
    T2aa_dag = FOwvfxn_aa.transpose(2,3,0,1)
    T2bbdag = FOwvfxn_bb.transpose(2,3,0,1)
    T2abdag = FOwvfxn_ab.transpose(2,3,0,1)
    T2aa    = pucc_t2aa
    T2bb    = pucc_t2bb
    T2ab    = pucc_t2ab
    resid_aa = secondOrder_residaa(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    resid_bb = secondOrder_residbb(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    resid_ab = secondOrder_residab(W_aaaa[oa,oa,va,va],W_bbbb[ob,ob,vb,vb],W_abab[oa,ob,va,vb],T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb)
    store_wavefxn_cap(driveCCobj,resid_aa,resid_bb,resid_ab,wvfxnC,prefactor=0.5) #     # Now build/store wavefunction and caps

    # Now, contract each of these w/ a Wn to build the lowest-order energy correction
    Wncap_diagA_E = get_energy(wvfxnA["t2aa"],wvfxnA["t2bb"],wvfxnA["t2ab"],W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob])
    Wncap_diagB_E = get_energy(wvfxnB["t2aa"],wvfxnB["t2bb"],wvfxnB["t2ab"],W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob])
    Wncap_diagC_E = get_energy(wvfxnC["t2aa"],wvfxnC["t2bb"],wvfxnC["t2ab"],W_aaaa[va,va,oa,oa],W_bbbb[vb,vb,ob,ob],W_abab[va,vb,oa,ob])
    print('Total E(3) contribution from <0|  WnQ2 @@ T2new |0>:', Wncap_diagA_E+Wncap_diagB_E + Wncap_diagC_E)

    # Now, contract each wavefuncion w/ a FO t2^ to build the higher-order energy correction
    FOwvfxn_aaDag = (FOwvfxn_VpUCC["t2aa"]*np.reciprocal(driveCCobj.denomInfo["D2aabkup"])).transpose(2,3,0,1) #FOwvfxn_aa*np.reciprocal(driveCCobj.denomInfo["D2aabkup"])).transpose(2,3,0,1)
    FOwvfxn_bbDag = (FOwvfxn_VpUCC["t2bb"]*np.reciprocal(driveCCobj.denomInfo["D2bbbkup"])).transpose(2,3,0,1) #(FOwvfxn_bb*np.reciprocal(driveCCobj.denomInfo["D2bbbkup"])).transpose(2,3,0,1)
    FOwvfxn_abDag = (FOwvfxn_VpUCC["t2ab"]*np.reciprocal(driveCCobj.denomInfo["D2abbkup"])).transpose(2,3,0,1) #(FOwvfxn_ab*np.reciprocal(driveCCobj.denomInfo["D2abbkup"])).transpose(2,3,0,1)
    pUCCWncap_diagA_E = get_energy(wvfxnA["t2aa"],wvfxnA["t2bb"],wvfxnA["t2ab"],FOwvfxn_aaDag,FOwvfxn_bbDag,FOwvfxn_abDag)
    pUCCWncap_diagB_E = get_energy(wvfxnB["t2aa"],wvfxnB["t2bb"],wvfxnB["t2ab"],FOwvfxn_aaDag,FOwvfxn_bbDag,FOwvfxn_abDag)
    pUCCWncap_diagC_E = get_energy(wvfxnC["t2aa"],wvfxnC["t2bb"],wvfxnC["t2ab"],FOwvfxn_aaDag,FOwvfxn_bbDag,FOwvfxn_abDag)
    print('Total E(3) contribution from T2^FO (T2^3) | 0> ',pUCCWncap_diagA_E+pUCCWncap_diagB_E+ pUCCWncap_diagC_E)
    E3=Wncap_diagA_E+Wncap_diagB_E + Wncap_diagC_E+pUCCWncap_diagA_E+pUCCWncap_diagB_E+ pUCCWncap_diagC_E
    return E3

def get_energy(t2aa,t2bb,t2ab,cap_aa,cap_bb,cap_ab):
    # Determine energy contribution
    energyA = 0.250000000 * np.einsum("abij,ijab->",cap_aa,t2aa,optimize="optimal")
    energyB = 1.000000000 * np.einsum("aAiI,iIaA->",cap_ab, t2ab,optimize="optimal")
    energyC = 0.250000000 * np.einsum("ABIJ,IJAB->",cap_bb,t2bb,optimize="optimal")

    print('all alpha/beta, and mixed energy contribution:', energyA, energyB, energyC)
    print('Total energy contribution:', energyA+energyB+energyC)
    return energyA+energyB+energyC

def store_wavefxn_cap(driveCCobj,resid_aa,resid_bb,resid_ab,save_dic,prefactor=1.0):
    resid_aa = tamps.antisym_T2(resid_aa,None,None)
    resid_bb = tamps.antisym_T2(resid_bb,None,None)

    # Ensure the 2nd-order caps aren't overwritten
    cap_aa = deepcopy(resid_aa).transpose(2,3,0,1)
    cap_bb = deepcopy(resid_bb).transpose(2,3,0,1)
    cap_ab = deepcopy(resid_ab).transpose(2,3,0,1)
    cap_ab = misc.zeroT2_offDiagonal(cap_ab)

    # Now, contract all of the bases against D2, then zero-out RoOvV, cap_ab
    resid_aa=resid_aa*driveCCobj.denomInfo["D2aabkup"]
    resid_bb = resid_bb*driveCCobj.denomInfo["D2bbbkup"]
    resid_ab =  resid_ab*driveCCobj.denomInfo["D2abbkup"]
    resid_ab = misc.zeroT2_offDiagonal(resid_ab)

    # build dictionary storing both the wavefunction and its corresponding "cap"
    save_dic.update({'t2aa':prefactor*resid_aa,'t2bb':prefactor*resid_bb,'t2ab':prefactor*resid_ab,
                               'D2T2aa_cap':cap_aa,'D2T2bb_cap':cap_bb,'D2T2ab_cap':cap_ab})



def secondOrder_residaa(W_aaaa,W_bbbb,W_abab,T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb):
    Roovv = -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += -0.500000000 * np.einsum("ikab,jIcA,cAkI->ijab",W_aaaa,T2ab,T2abdag,optimize="optimal")
    Roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += -0.500000000 * np.einsum("ijac,kIbA,cAkI->ijab",W_aaaa,T2ab,T2abdag,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("ikac,jlbd,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("ikac,jIbA,cAkI->ijab",W_aaaa,T2ab,T2abdag,optimize="optimal")
    Roovv += -0.250000000 * np.einsum("klac,ijbd,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("iIaA,jkbc,cAkI->ijab",W_abab,T2aa,T2abdag,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("iIaA,jJbB,ABIJ->ijab",W_abab,T2ab,T2bbdag,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("kIaA,ijbc,cAkI->ijab",W_abab,T2aa,T2abdag,optimize="optimal")
    Roovv += 0.062500000 * np.einsum("ijcd,klab,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += -0.250000000 * np.einsum("ikcd,jlab,cdkl->ijab",W_aaaa,T2aa,T2aa_dag,optimize="optimal")
    Roovv += 0.500000000 * np.einsum("iIcA,jkab,cAkI->ijab",W_abab,T2aa,T2abdag,optimize="optimal")
    return Roovv

def secondOrder_residbb(W_aaaa,W_bbbb,W_abab,T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb):
    T2ab_dag=T2abdag
    T2bb_dag=T2bbdag
    ROOVV = 1.000000000 * np.einsum("iIaA,jJbB,abij->IJAB",W_abab,T2ab,T2aa_dag,optimize="optimal")
    ROOVV += 1.000000000 * np.einsum("iIaA,JKBC,aCiK->IJAB",W_abab,T2bb,T2ab_dag,optimize="optimal")
    ROOVV += 0.500000000 * np.einsum("iKaA,IJBC,aCiK->IJAB",W_abab,T2bb,T2ab_dag,optimize="optimal")
    ROOVV += 0.500000000 * np.einsum("iIaC,JKAB,aCiK->IJAB",W_abab,T2bb,T2ab_dag,optimize="optimal")
    ROOVV += -0.500000000 * np.einsum("IKAB,iJaC,aCiK->IJAB",W_bbbb,T2ab,T2ab_dag,optimize="optimal")
    ROOVV += -0.250000000 * np.einsum("IKAB,JLCD,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += 0.062500000 * np.einsum("KLAB,IJCD,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += -0.500000000 * np.einsum("IJAC,iKaB,aCiK->IJAB",W_bbbb,T2ab,T2ab_dag,optimize="optimal")
    ROOVV += -0.250000000 * np.einsum("IJAC,KLBD,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += 1.000000000 * np.einsum("IKAC,iJaB,aCiK->IJAB",W_bbbb,T2ab,T2ab_dag,optimize="optimal")
    ROOVV += 1.000000000 * np.einsum("IKAC,JLBD,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += -0.250000000 * np.einsum("KLAC,IJBD,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += 0.062500000 * np.einsum("IJCD,KLAB,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    ROOVV += -0.250000000 * np.einsum("IKCD,JLAB,CDKL->IJAB",W_bbbb,T2bb,T2bb_dag,optimize="optimal")
    return ROOVV

def secondOrder_residab(W_aaaa,W_bbbb,W_abab,T2aa,T2bb,T2ab,T2aa_dag,T2abdag,T2bbdag,oa,ob,va,vb):
    RoOvV = 1.000000000 * np.einsum("ijab,kIcA,bcjk->iIaA",W_aaaa,T2ab,T2aa_dag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("ijab,IJAB,bBjJ->iIaA",W_aaaa,T2bb,T2abdag,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("jkab,iIcA,bcjk->iIaA",W_aaaa,T2ab,T2aa_dag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iJaA,jIbB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("iJaA,IKBC,BCJK->iIaA",W_abab,T2bb,T2bbdag,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("jIaA,ikbc,bcjk->iIaA",W_abab,T2aa,T2aa_dag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("jIaA,iJbB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("jJaA,iIbB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iIaB,jJbA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("iIaB,JKAC,BCJK->iIaA",W_abab,T2bb,T2bbdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iJaB,jIbA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iJaB,IKAC,BCJK->iIaA",W_abab,T2bb,T2bbdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("jIaB,iJbA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("jJaB,iIbA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("ijbc,kIaA,bcjk->iIaA",W_aaaa,T2ab,T2aa_dag,optimize="optimal")
    RoOvV += -0.500000000 * np.einsum("iIbA,jkac,bcjk->iIaA",W_abab,T2aa,T2aa_dag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iIbA,jJaB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iJbA,jIaB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("jIbA,ikac,bcjk->iIaA",W_abab,T2aa,T2aa_dag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("jIbA,iJaB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("jJbA,iIaB,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIbB,jJaA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iJbB,jIaA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("jIbB,iJaA,bBjJ->iIaA",W_abab,T2ab,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("IJAB,ijab,bBjJ->iIaA",W_bbbb,T2aa,T2abdag,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("IJAB,iKaC,BCJK->iIaA",W_bbbb,T2ab,T2bbdag,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("JKAB,iIaC,BCJK->iIaA",W_bbbb,T2ab,T2bbdag,optimize="optimal")
    RoOvV += 0.500000000 * np.einsum("IJBC,iKaA,BCJK->iIaA",W_bbbb,T2ab,T2bbdag,optimize="optimal")
    return RoOvV

def VdotFOwvfxn_resid_aa(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb):
    Roovv += 0.125000000 * np.einsum("ijkl,klab->ijab",W_aaaa[oa,oa,oa,oa],T2_aa,optimize="optimal")
    Roovv += -1.000000000 * np.einsum("icka,jkbc->ijab",W_aaaa[oa,va,oa,va],T2_aa,optimize="optimal")
    Roovv += 0.125000000 * np.einsum("cdab,ijcd->ijab",W_aaaa[va,va,va,va],T2_aa,optimize="optimal")
    Roovv += 1.000000000 * np.einsum("iAaI,jIbA->ijab",W_abab[oa,vb,va,ob],T2_ab,optimize="optimal")
    nocc = nvir = None
    Roovv = tamps.antisym_T2(Roovv,nocc,nvir)

    return Roovv

def VdotFOwvfxn_resid_bb(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb):
    ROOVV = 1.000000000 * np.einsum("aIiA,iJaB->IJAB",W_abab[va,ob,oa,vb],T2_ab,optimize="optimal")
    ROOVV += 0.125000000 * np.einsum("IJKL,KLAB->IJAB",W_bbbb[ob,ob,ob,ob],T2_bb,optimize="optimal")
    ROOVV += -1.000000000 * np.einsum("ICKA,JKBC->IJAB",W_bbbb[ob,vb,ob,vb],T2_bb,optimize="optimal")
    ROOVV += 0.125000000 * np.einsum("CDAB,IJCD->IJAB",W_bbbb[vb,vb,vb,vb],T2_bb,optimize="optimal")
    nocc = nvir = None
    ROOVV = tamps.antisym_T2(ROOVV,nocc,nvir)
    return ROOVV



def VdotFOwvfxn_resid_ab(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb):
    RoOvV = -1.000000000 * np.einsum("ibja,jIbA->iIaA",W_aaaa[oa,va,oa,va],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iIjJ,jJaA->iIaA",W_abab[oa,ob,oa,ob],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("iBjA,jIaB->iIaA",W_abab[oa,vb,oa,vb],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bIjA,ijab->iIaA",W_abab[va,ob,oa,vb],T2_aa,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("iBaJ,IJAB->iIaA",W_abab[oa,vb,va,ob],T2_bb,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("bIaJ,iJbA->iIaA",W_abab[va,ob,va,ob],T2_ab,optimize="optimal")
    RoOvV += 1.000000000 * np.einsum("bBaA,iIbB->iIaA",W_abab[va,vb,va,vb],T2_ab,optimize="optimal")
    RoOvV += -1.000000000 * np.einsum("IBJA,iJaB->iIaA",W_bbbb[ob,vb,ob,vb],T2_ab,optimize="optimal")
    #nocc = nvir = None
    #RoOvV = tamps.antisym_T2(RoOvV,nocc,nvir)
    return RoOvV
