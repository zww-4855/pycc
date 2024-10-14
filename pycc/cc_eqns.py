import numpy as np
import pycc.tamps as tamps
import pycc.ccdq_eqns as ccdq_eqns
import pycc.ccsdt_eqns as ccsdt_eqns
import pycc.ucc_eqns as ucc_eqns
import pycc.spin_integratedUCCeqns as SIucc_eqns
import pycc.misc as misc

def cceqns_driver(driveCCobj,cc_info):

    if "fastSIcalc" in cc_info: # spin-integrated code
        nocc_aa = driveCCobj.occInfo["nocc_aa"]
        nocc_bb = driveCCobj.occInfo["nocc_bb"]
        nvirt_aa = driveCCobj.occInfo["nvirt_aa"]
        nvirt_bb = driveCCobj.occInfo["nvirt_bb"]

        oa = driveCCobj.occSliceInfo["occ_aa"]
        ob = driveCCobj.occSliceInfo["occ_bb"]
        va = driveCCobj.occSliceInfo["virt_aa"]
        vb = driveCCobj.occSliceInfo["virt_bb"]


        W_aaaa = driveCCobj.integralInfo["tei_aaaa"]
        W_bbbb = driveCCobj.integralInfo["tei_bbbb"]
        W_abab = driveCCobj.integralInfo["tei_abab"]

        Fock_aa = driveCCobj.integralInfo["oei_aa"]
        Fock_bb = driveCCobj.integralInfo["oei_bb"]

        D2aa = driveCCobj.denomInfo["D2aa"]
        D2bb = driveCCobj.denomInfo["D2bb"]
        D2ab = driveCCobj.denomInfo["D2ab"]
        T2_aa = driveCCobj.tamps["t2aa"]
        T2_bb = driveCCobj.tamps["t2bb"]
        T2_ab = driveCCobj.tamps["t2ab"]
       
        resid_t2aa = SIucc_eqns.UCC3_t2resid_aa(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock_aa)
        resid_t2bb = SIucc_eqns.UCC3_t2resid_bb(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock_bb)
        resid_t2ab = SIucc_eqns.UCC3_t2resid_ab(W_aaaa,W_bbbb,W_abab,T2_aa,T2_bb,T2_ab,oa,ob,va,vb,Fock_aa,Fock_bb) 
     
        resid_t2aa += np.reciprocal(driveCCobj.denomInfo["D2aa"])*T2_aa
        resid_t2bb += np.reciprocal(driveCCobj.denomInfo["D2bb"])*T2_bb
        resid_t2ab += np.reciprocal(driveCCobj.denomInfo["D2ab"])*T2_ab

        if "pCCD" in driveCCobj.cc_type or "pLCCD" in driveCCobj.cc_type: # zero the off-diagonal if pCCD/pLCCD calc
            resid_t2aa = resid_t2bb = 0.0*resid_t2aa
            resid_t2ab = misc.zeroT2_offDiagonal(resid_t2ab)

        driveCCobj.tamps.update({'t2aa':resid_t2aa*D2aa,"t2bb":resid_t2bb*D2bb,"t2ab":resid_t2ab*D2ab})

    if "slowSOcalc" in cc_info: # spin-orb eqns
        o=driveCCobj.occSliceInfo["occ_aa"]
        v=driveCCobj.occSliceInfo["virt_aa"]
    
        nocc=driveCCobj.occInfo["nocc_aa"]
        nvirt=driveCCobj.occInfo["nvirt_aa"]
    
        W=driveCCobj.integralInfo["tei"]
        Fock=driveCCobj.integralInfo["oei"]

        if "CCSDT" in cc_info["slowSOcalc"]: # Generate CCSDT resid eqns
            print('Entering CCSDT eqns')
            T1=driveCCobj.tamps["t1aa"]
            T2=driveCCobj.tamps["t2aa"]
            T3=driveCCobj.tamps["t3aa"]
            D1=driveCCobj.denomInfo["D1aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            D3=driveCCobj.denomInfo["D3aa"]

            resid_t1=ccsdt_eqns.ccsdt_t1eqns(Fock,W,T1,T2,T3,o,v)
            resid_t2=ccsdt_eqns.ccsdt_t2eqns(Fock,W,T1,T2,T3,o,v,driveCCobj)
            resid_t3=ccsdt_eqns.ccsdt_t3eqns(Fock,W,T1,T2,T3,o,v)

            resid_t1+=np.reciprocal(driveCCobj.denomInfo["D1aa"])*T1
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])*T2
            resid_t3+=np.reciprocal(driveCCobj.denomInfo["D3aa"])*T3

            driveCCobj.tamps.update({'t1aa':resid_t1*D1,"t2aa":resid_t2*D2,"t3aa":resid_t3*D3})

        elif "UCC" in cc_info["slowSOcalc"]:
            print('Entering UCC eqns')
            T2=driveCCobj.tamps["t2aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            no=np.shape(T2)[0]
            nv=np.shape(T2)[2]
            T1=driveCCobj.tamps.get("t1aa",np.zeros((no,nv)))
            print('T1:',T1)
            D1=driveCCobj.denomInfo.get("D1aa",None) # replace w/ string 'zero' if no D1
            resid_t1,resid_t2 = ucc_eqns.ucc_eqnDriver(cc_info["slowSOcalc"],Fock,W,T1,T2,o,v)

            resid_t2 += np.reciprocal(D2)*T2
            driveCCobj.tamps.update({"t2aa":resid_t2*D2})
            if D1 is None:
                driveCCobj.tamps.update({"t1aa":np.zeros((no,nv))})
            else:
                resid_t1 += np.reciprocal(D1)*T1
                driveCCobj.tamps.update({"t1aa":resid_t1*D1})

        elif "CCSD" in cc_info["slowSOcalc"]: #Generate CCSD resid eqns
            print('Entering CCSD eqns')
            T1=driveCCobj.tamps["t1aa"]
            T2=driveCCobj.tamps["t2aa"]
            D1=driveCCobj.denomInfo["D1aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            resid_t1=ccsd_t1resid(Fock,W,T1,T2,o,v)
            resid_t2=ccsd_t2resid(Fock,W,T1,T2,o,v,nocc,nvirt)
            resid_t1+=np.reciprocal(driveCCobj.denomInfo["D1aa"])*T1
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])*T2
            driveCCobj.tamps.update({"t1aa":resid_t1*D1,"t2aa":resid_t2*D2})

        elif "CCDQ" in cc_info["slowSOcalc"]: # Generate CCDQ resid eqns
            print('Entering CCDQ eqns')
            T2=driveCCobj.tamps["t2aa"]
            T4=driveCCobj.tamps["t4aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            D4=driveCCobj.denomInfo["D4aa"]
            resid_t2=ccdq_eqns.ccdq_t2eqns(Fock,W,T2,T4,o,v)
            resid_t4=ccdq_eqns.ccdq_t4eqns(Fock,W,T2,T4,o,v)
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])*T2
            resid_t4+=np.reciprocal(driveCCobj.denomInfo["D4aa"])*T4
            driveCCobj.tamps.update({"t2aa":resid_t2*D2,"t4aa":resid_t4*D4})




        elif "D" in cc_info["slowSOcalc"]: #Generate CCD base resids
            T2=driveCCobj.tamps["t2aa"]
            D2=driveCCobj.denomInfo["D2aa"]
            resid_t2=ccsd_t2resid(Fock,W,np.zeros((nocc,nvirt)),T2,o,v,nocc,nvirt)
            resid_t2+=np.reciprocal(driveCCobj.denomInfo["D2aa"])*T2
            driveCCobj.tamps.update({"t2aa":resid_t2*D2})

def ccsd_t1resid(F,W,T1,T2,o,v):
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
    return rov


def ccsd_t2resid(F,W,T1,T2,o,v,nocc,nvir):
    # contributions to the residual
    roovv = 0.500000000 * np.einsum("ik,jkab->ijab",F[o,o],T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ka,ijbc->ijab",F[v,o],T1,T2,optimize="optimal")
    roovv += 0.500000000 * np.einsum("ck,ic,jkab->ijab",F[v,o],T1,T2,optimize="optimal")
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
    roovv += -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ic,jd,klab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ic,jd,cdab->ijab",T1,T1,W[v,v,v,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,kd,jlab,cdkl->ijab",T1,T1,T2,W[v,v,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ic,klab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 1.000000000 * np.einsum("ic,jkad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ilab,jckl->ijab",T1,T2,W[o,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("kc,ijad,cdkb->ijab",T1,T2,W[v,v,o,v],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.125000000 * np.einsum("klab,ijkl->ijab",T2,W[o,o,o,o],optimize="optimal")
    roovv += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W[v,v,o,o],optimize="optimal")
    roovv += -1.000000000 * np.einsum("ikac,jckb->ijab",T2,W[o,v,o,v],optimize="optimal")
    roovv += 0.125000000 * np.einsum("ijcd,cdab->ijab",T2,W[v,v,v,v],optimize="optimal")
    roovv += 0.250000000 * np.einsum("ijab->ijab",W[o,o,v,v],optimize="optimal")

    roovv=tamps.antisym_T2(roovv,nocc,nvir)

    return roovv


