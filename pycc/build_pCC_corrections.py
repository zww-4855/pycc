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
    FO_diagramA(driveCCobj,W_aaaa,W_bbbb,W_abab,oa,ob,va,vb)
    FO_diagramB(driveCCobj,W_aaaa,W_bbbb,W_abab,driveCCobj.tamps["t2ab"],oa,ob,va,vb)
    FO_diagramD(driveCCobj,W_aaaa,W_bbbb,W_abab,T2_ab,oa,ob,va,vb)


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
