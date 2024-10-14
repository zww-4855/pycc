import numpy as np
import pycc.set_denoms as set_denoms
import pycc.tamps as tamps
import pycc.cc_energy as cc_energy
import pycc.cc_eqns as cc_eqns
import pycc.rdm1 as rdm1
import pycc.props as props
import pycc.misc as misc
import pickle

class SetupCC():
    """ SetupCC() is a class that holds all data necessary for a CC calculation. Similar in design to C-struct
    :param pyscf_mf: Mean-field object from PySCF

    :param pyscf_mol: Molecule object from PySCF

    :param cc_info: Dictionary that stores all user-defined parameters that customize a CC calculation

    :return: An object that contains information harvested at the mean-field level from PySCF, for subsequent use within pycc. 

    """
    def __init__(self,pyscf_mf,pyscf_mol,cc_info):#Set the defaults up for CC calculation
        """
         Extracts the pyscf_mf, pyscf_mol, and relevant user-input from the cc_info dictionary for use in a subsequent pycc CC calculation
        """
        
        # Load mean-field information from PySCF object
        self.hf_e=pyscf_mf.e_tot
        self.nuc_e=pyscf_mf.energy_nuc()
        
        # Initialize basics of CC calculation
        self.max_iter=cc_info.get("max_iter",100)
        self.dump_tamps=cc_info.get('dump_tamps',False)
        self.dropcore=cc_info.get('dropcore',0)
        self.stopping_eps=cc_info.get("stopping_eps",10**-8)
        self.diis_size=cc_info.get("diis_size")
        self.diis_start_cycle=cc_info.get("diis_start_cycle")

        # Initialize data dictionaries
        self.occInfo=None
        self.occSliceInfo=None
        self.denomInfo={}
        self.integralInfo={}
        self.eps=None
        self.nmo = np.shape(pyscf_mf.mo_coeff)[0]

        if "slowSOcalc" in cc_info: # If a slow, spin-orb-based CC calc
            self.cc_calcs=cc_info.get("slowSOcalc",'CCD')
            self.get_occInfo(pyscf_mf)
            self.get_integrals(pyscf_mf,pyscf_mol)
            self.get_denomsSlow(pyscf_mf,cc_info["slowSOcalc"])
        elif "fastSIcalc" in cc_info: # Initial spin integrated code
            self.cc_calcs=cc_info.get("fastSIcalc",'LCCD')
            self.get_occInfo(pyscf_mf,'fastSIcalc')
            self.get_spinIntegrated_integrals(pyscf_mf,pyscf_mol)
            self.get_denomsFast(pyscf_mf,cc_info["fastSIcalc"])

        ## TO DO:: ADD OPTION FOR (SPIN-INTEGRATED ? ) CC EQNS USING INTERMEDIATES, AND INTERFACE TO XACC


    def get_denomsFast(self,pyscf_mf,cc_calc):
        virt_aa=self.occSliceInfo["virt_aa"]
        virt_bb=self.occSliceInfo["virt_bb"]
        occ_aa=self.occSliceInfo["occ_aa"]
        occ_bb=self.occSliceInfo["occ_bb"]

        epsaa=self.eps["eps_aa"]
        epsbb=self.eps["eps_bb"]
        n = np.newaxis

        if "S" in cc_calc:
            D1aa,D1bb=set_denoms.D1denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n)
            self.denomInfo.update({"D1aa":D1aa,"D1bb":D1bb})
        if "D" in cc_calc:
            D2aa,D2bb,D2ab=set_denoms.D2denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n)
            self.denomInfo.update({"D2aa":D2aa,"D2bb":D2bb,"D2ab":D2ab})

        if "pCCD" in cc_calc or "pLCCD" in cc_calc:
            D2aa = D2bb = 0.0*D2aa
            D2ab = misc.zeroT2_offDiagonal(D2ab)

    def get_denomsSlow(self,pyscf_mf,cc_calc):
        virt_aa=self.occSliceInfo["virt_aa"]
        occ_aa=self.occSliceInfo["occ_aa"]
        epsaa=self.eps
        n = np.newaxis
        if "S" in cc_calc: # Get T1 denoms
            self.denomInfo.update({"D1aa":set_denoms.D1denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "D" in cc_calc: # Get T2 denoms
            self.denomInfo.update({"D2aa":set_denoms.D2denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "T" in cc_calc: #Get T3 denoms
            self.denomInfo.update({"D3aa":set_denoms.D3denomSlow(epsaa,occ_aa,virt_aa,n)})
        if "UT2" in cc_calc or "X" in cc_calc or "Qdebug" in cc_calc or "Q" in cc_calc:
            self.denomInfo.update({"D4aa":set_denoms.D4denomSlow(epsaa,occ_aa,virt_aa,n)})


    def spin_block_tei(self,I):
        """
        Function that spin blocks two-electron integrals
        Using np.kron, we project I into the space of the 2x2 identity, tranpose the result
        and project into the space of the 2x2 identity again. This doubles the size of each axis.
        The result is our two electron integral tensor in the spin orbital form.
        """
        identity = np.eye(2)
        I = np.kron(identity, I)
        return np.kron(identity, I.T)


    def get_spinIntegrated_integrals(self,pyscf_mf,pyscf_mol):
        """
        Converts UHF/RHF 1 and 2e- integrals into the MO framework. For use in RHF/UHF codes ONLY.
        
        :param mf: PySCF SCF object
        :param mol: PySCF Molecule object
        :param h1e: Core Hamiltonian
        :param f: Fock Matrix
        :param na: Number of alpha occupied orbitals
        :param nb: Number of beta occupied orbitals
        :param orb: SCF coefficients
         
        :return: Returns MO-transformed fock and two electron integrals in a dictionary that is accessible using keys "oei" and "tei", respectively. 
        
        """
        h1e = np.array((pyscf_mf.get_hcore(), pyscf_mf.get_hcore()))
        f = pyscf_mf.get_fock()
        orb = pyscf_mf.mo_coeff
        na, nb = pyscf_mf.nelec

        h1aa = orb[0].T @ h1e[0] @ orb[0]
        h1bb = orb[1].T @ h1e[1] @ orb[1]
    
        # f=mf.get_fock()
        faa = orb[0].T @ f[0] @ orb[0]
        fbb = orb[1].T @ f[1] @ orb[1]
    
        eri = pyscf_mol.intor("int2e", aosym="s1")
        eri = pyscf_mol.intor('int2e',aosym='s1')
        if np.shape(eri)==(0,0,0,0):# otherwise,
            norbs=orb[0].shape[0]
            print(norbs)#.shape[0])
            eri=np.zeros((norbs,norbs,norbs,norbs))
            print(np.shape(eri))
            with open('ao_tei.pickle', 'rb') as handle:
                eri=pickle.load(handle)
        else:
            from pyscf import ao2mo
            g_aaaa = ao2mo.incore.general(eri, (orb[0], orb[0], orb[0], orb[0]))
            g_bbbb = ao2mo.incore.general(eri, (orb[1], orb[1], orb[1], orb[1]))
            g_abab = ao2mo.incore.general(eri, (orb[0], orb[0], orb[1], orb[1]))
    
        # Verify the 2e- integral coulomb energy
        ga = g_aaaa.transpose(0, 2, 1, 3)
        gb = g_bbbb.transpose(0, 2, 1, 3)
        e_coul = np.einsum("ijij", ga[:na, :na, :na, :na]) + np.einsum(
            "ijij", gb[:nb, :nb, :nb, :nb]
        )
        e_exch = 0.5 * np.einsum("ijji", ga[:na, :na, :na, :na]) + 0.5 * np.einsum(
            "ijji", gb[:nb, :nb, :nb, :nb]
        )
    
        print("total 2e- integral energy:", e_coul - e_exch)
    
        # Now, convert to Dirac notation, and antisymmetrize g_aaaa/g_bbbb
        g_aaaa = g_aaaa.transpose(0, 2, 1, 3) - g_aaaa.transpose(0, 3, 2, 1)  # (0,3,1,2)
        g_bbbb = g_bbbb.transpose(0, 2, 1, 3) - g_bbbb.transpose(0, 3, 2, 1)
        g_abab = g_abab.transpose(0, 2, 1, 3)
        print(np.shape(g_aaaa))
    
        # Now, verify the UHF energy
        e1 = 0.5 * np.einsum("ii", h1aa[:na, :na]) + 0.5 * np.einsum("ii", h1bb[:nb, :nb])
        e2 = 0.5 * np.einsum("ii", faa[:na, :na]) + 0.5 * np.einsum("ii", fbb[:nb, :nb])
        totSCFenergy = e1 + e2 + pyscf_mf.energy_nuc()
        print("final rhf/uhf energy:", totSCFenergy)
        
        # set the spin-integrated integrals and mo_energies
        self.integralInfo={"oei_aa":faa,"oei_bb":fbb,"tei_aaaa":g_aaaa,"tei_bbbb":g_bbbb,"tei_abab":g_abab}
        self.eps = {"eps_aa":pyscf_mf.mo_energy[0],"eps_bb":pyscf_mf.mo_energy[1]}


    def get_integrals(self,pyscf_mf,pyscf_mol):
        dropcore=self.dropcore
        print('dropcore:',dropcore)
        if 'RHF' in str(type(pyscf_mf)): # running RHF calculation
            Ca = Cb = np.asarray(pyscf_mf.mo_coeff)
            eps_a = eps_b = np.asarray(pyscf_mf.mo_energy)
    
        elif 'UHF' in str(type(pyscf_mf)): # running UHF calculation
            Ca = np.asarray(pyscf_mf.mo_coeff[0])
            Cb = np.asarray(pyscf_mf.mo_coeff[1])
            eps_a = np.asarray(pyscf_mf.mo_energy[0])
            eps_b = np.asarray(pyscf_mf.mo_energy[1])
            print('eps_a',eps_a)


        # default is to try and use PySCF object to harvest AO 2eints
        eri = pyscf_mol.intor('int2e',aosym='s1')
        if np.shape(eri)==(0,0,0,0):# otherwise,
            norbs = pyscf_mf.get_fock().shape[0]
            eri=np.zeros((norbs,norbs,norbs,norbs))
            print(np.shape(eri))
            with open('ao_tei.pickle', 'rb') as handle:
                eri=pickle.load(handle)
    
        C = np.block([
                 [      Ca           ,   np.zeros_like(Cb) ],
                 [np.zeros_like(Ca)  ,          Cb         ]
                ])
    
    
        I = np.asarray(eri)
        I_spinblock = self.spin_block_tei(I)
        # Converts chemist's notation to physicist's notation, and antisymmetrize
        # (pq | rs) ---> <pr | qs>
        # Physicist's notation
        tmp = I_spinblock.transpose(0, 2, 1, 3)
        # Antisymmetrize:
        # <pr||qs> = <pr | qs> - <pr | sq>
        gao = tmp - tmp.transpose(0, 1, 3, 2)
        eps = np.append(eps_a, eps_b)
    
        # Sort the columns of C according to the order of increasing orbital energies
        C = C[:, eps.argsort()[dropcore*2:]]
        # Sort orbital energies in increasing order
        eps = np.sort(eps)[dropcore*2:]
        self.eps=eps

            # Transform gao, which is the spin-blocked 4d array of physicist's notation,
    # antisymmetric two-electron integrals, into the MO basis using MO coefficients
        gmo = np.einsum('pQRS, pP -> PQRS',
              np.einsum('pqRS, qQ -> pQRS',
              np.einsum('pqrS, rR -> pqRS',
              np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

        fock=np.diag(eps)
        self.integralInfo={"oei":fock,"tei":gmo}



    def get_occInfo(self,pyscf_mf,calcType=None):
        dropcore=self.dropcore
        print('dropcore:',dropcore)
        print('calctype:',calcType)
        if calcType=="fastSIcalc": # running UHF, spin-integrated calc
            print('inside get_occ info:')
            na, nb = pyscf_mf.nelec
            f = pyscf_mf.get_fock()
            nvirta = f[0].shape[0] - na
            nvirtb = f[1].shape[0] - nb
            self.occInfo={"nocc_aa":na,"nocc_bb":nb,"nvirt_aa":nvirta,"nvirt_bb":nvirtb}

            occ_aa = slice(None, na)
            virt_aa = slice(na, None)
            occ_bb = slice(None, nb)
            virt_bb = slice(nb, None)
            self.occSliceInfo={"occ_aa":  occ_aa, "virt_aa":virt_aa,
                             "occ_bb": occ_bb, "virt_bb":virt_bb}
            return

        elif 'RHF' in str(type(pyscf_mf)): # running RHF, spin-orbital-based calculation
            if dropcore>0: 
                print('dropcore not implemented for RHF')
                sys.exit()
            occ = pyscf_mf.mo_occ
            nele = int(sum(occ))
            nocc = nele // 2
            norbs = pyscf_mf.get_fock().shape[0] #oei.shape[0]
            nsvirt = 2 * (norbs - nocc)
            nsocc = 2 * nocc
            self.occInfo={"nocc_aa":nsocc,"nvirt_aa":nsvirt}

        elif 'UHF' in str(type(pyscf_mf)): # running UHF calculation
            norbs=pyscf_mf.get_fock().shape[0] + pyscf_mf.get_fock().shape[1]
            na,nb=pyscf_mf.nelec
            nele=na+nb-2*dropcore
            nsvirt = (pyscf_mf.get_fock().shape[1]-na)+(pyscf_mf.get_fock().shape[2]-nb)# #2 * (pyscf_mf.get_fock()[0].shape[0] - na)#(norbs - nocc)
            self.occInfo={"nocc_aa":nele,"nvirt_aa":nsvirt}
    
    
        n = np.newaxis
        o = slice(None, nele)
        v = slice(nele, None)
        self.occSliceInfo={"occ_aa":o,"virt_aa":v}




class DriveCC(SetupCC):
    def __init__(self,pyscf_mf,pyscf_mol,cc_info,t2ampFile=None):
        SetupCC.__init__(self,pyscf_mf,pyscf_mol,cc_info)
        print(self.cc_calcs)
        self.correlationE={} #options: totalCorrCorrection, and options for (T), (Qf), etc
        self.tamps={}     #TO DO:: Load T2 if not None
        self.rdm1={}

        # setup t amplitudes TODO:: ADD in query/setup for tamps of spin-intgr eqns
        if "slowSOcalc" in cc_info:
            nocc=self.occInfo["nocc_aa"]
            nvirt=self.occInfo["nvirt_aa"]
            o=self.occSliceInfo["occ_aa"]
            v=self.occSliceInfo["virt_aa"]
            self.cc_type = cc_info["slowSOcalc"]
            self.tamps = tamps.set_tampsSLOW(cc_info["slowSOcalc"],nocc,nvirt,self.integralInfo["tei"][o,o,v,v]*self.denomInfo["D2aa"],t2ampFile) 

        elif "fastSIcalc" in cc_info:
            nocc_aa = self.occInfo["nocc_aa"]
            nocc_bb = self.occInfo["nocc_bb"]
            nvirt_aa = self.occInfo["nvirt_aa"]
            nvirt_bb = self.occInfo["nvirt_bb"]

            oa = self.occSliceInfo["occ_aa"]
            ob = self.occSliceInfo["occ_bb"]
            va = self.occSliceInfo["virt_aa"]
            vb = self.occSliceInfo["virt_bb"]

            self.cc_type = cc_info["fastSIcalc"]
            initT2_aaaa = self.integralInfo["tei_aaaa"][oa,oa,va,va]*self.denomInfo["D2aa"]
            initT2_bbbb = self.integralInfo["tei_bbbb"][ob,ob,vb,vb]*self.denomInfo["D2bb"]
            initT2_abab = self.integralInfo["tei_abab"][oa,ob,va,vb]*self.denomInfo["D2ab"]

            self.tamps = tamps.set_tampsFAST(self.cc_type,nocc_aa,nocc_bb,nvirt_aa,nvirt_bb,initT2_aaaa,initT2_bbbb,initT2_abab)

        if self.diis_size is not None:# only works for spin-orb models rn
            from pycc.diis import DIIS
            self.diis_update=DIIS(self.diis_size, start_iter=self.diis_start_cycle)
            self.old_vec=tamps.get_oldvec(self.tamps,self.cc_type)



    def kernel(self,cc_info):


        print("    ==> ", self.cc_type, " amplitude equations <==")
        print("")
        print("     Iter              Corr. Energy                 |dE|    ")
        print(flush=True)

        convergedCC=False
        old_energy=cc_energy.ccenergy_driver(self,cc_info)
        self.correlationE.update({"mp2e":old_energy})
        print('old energy:',old_energy-self.hf_e+self.nuc_e)
        for idx in range(self.max_iter):
            cc_eqns.cceqns_driver(self,cc_info)
            current_energy=cc_energy.ccenergy_driver(self,cc_info)
            delta_e = np.abs(old_energy - current_energy)
            self.correlationE.update({"totalCorrE":current_energy})#-self.hf_e})
            self.correlationE.update({"totalE":current_energy+self.hf_e})
            print(
                "    {: 5d} {: 20.12f} {: 20.12f} ".format(
                    idx, self.correlationE["totalCorrE"], delta_e
                )
            )
            print(flush=True)

            if delta_e < self.stopping_eps:  # and res_norm < stopping_eps:
                convergedCC=True
                break
            else:
                old_energy = current_energy
            if idx > self.max_iter:
                raise ValueError("CC iterations did not converge")

        if convergedCC:
            self.finalizeCC()
        else:
            print('CC iterations did not converge!!')
            sys.exit()
        


    def finalizeCC(self):

        print("\n\n\n")
        print("************************************************************")
        print("************************************************************\n\n")
        print('Total SCF energy: \t {: 20.12f}'.format(self.hf_e))
        print('Nuclear repulsion energy: \t {: 20.12f}'.format(self.nuc_e))
        print(f"\n \t**** Results for {self.cc_type}: **** \n\n")

        print('Total energy (SCF + correlation): \t {: 20.12f}'.format(self.correlationE["totalE"]))
        print("Iterative correlation energy: \t {: 20.12f}".format(self.correlationE["totalCorrE"]))
        print(flush=True)
        # Handle the post-hoc perturbative corrections, if there are any
        if "(" in self.cc_type:
            correctionDict = cc_energy.perturbE_driver(self, self.cc_type)
            for key in correctionDict.keys():
                print(key,"{: 20.12f}".format(correctionDict[key]))

        if self.dump_tamps: 
            with open('t2amps.pickle', 'wb') as f:
                pickle.dump(self.tamps["t2aa"], f)


    def drive_rdm(self,pyscf_mol,pyscf_mf):
        spin_rdm1 = rdm1.build_Spinrdm1(self)
        alpha_rdm,beta_rdm = rdm1.spin_to_spatial_rdm1(self,spin_rdm1)
        # get first-order props
        self.rdm1.update({"spin_rdm":spin_rdm1,"alpha":alpha_rdm,"beta":beta_rdm})   
        props.dipole_moment(pyscf_mol,pyscf_mf,alpha_rdm+beta_rdm,spin_rdm1)
  


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("Running bare pycc.py")
