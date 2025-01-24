import numpy as np
from math import floor
import pycc.set_denoms as set_denoms
import pycc.tamps as tamps
import pycc.cc_energy as cc_energy
import pycc.cc_eqns as cc_eqns
import pycc.rdm1 as rdm1
import pycc.props as props
import pycc.misc as misc
import pycc.build_pCC_corrections as build_pCC_corrections
import pycc.pcc_base as pcc_base
import pycc.build_sqrbrak_corrections as build_sqrbrak_corrections
import pycc.ucc_eqns as ucc_eqns
from copy import deepcopy
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
        # Initialize data dictionaries
        self.occInfo=None
        self.occSliceInfo=None
        self.denomInfo={}
        self.integralInfo={}
        self.eps=None

#        if isinstance(self,Run_xacc): #if Run_xacc class instatiated, do nothing
#            print('This is an instance of Run_xacc')
#            return
        # Load mean-field information from PySCF object
        self.hf_e=pyscf_mf.e_tot
        self.nuc_e=pyscf_mf.energy_nuc()
        self.nmo = np.shape(pyscf_mf.mo_coeff)[0]

        # Initialize basics of CC calculation
        self.max_iter=cc_info.get("max_iter",100)
        self.dump_tamps=cc_info.get('dump_tamps',False)
        self.dropcore=cc_info.get('dropcore',0)
        self.stopping_eps=cc_info.get("stopping_eps",10**-8)
        self.diis_size=cc_info.get("diis_size")
        self.diis_start_cycle=cc_info.get("diis_start_cycle")


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
            from copy import deepcopy
            D2aa_bk = deepcopy(D2aa)
            D2bb_bk = deepcopy(D2bb)
            D2ab_bk = deepcopy(D2ab)
            self.denomInfo.update({"D2aabkup":D2aa_bk,"D2bbbkup":D2bb_bk,"D2abbkup":D2ab_bk})
            D2aa = D2bb = 0.0*D2aa
            D2ab = misc.zeroT2_offDiagonal(D2ab)
            D1aa,D1bb = set_denoms.D1denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n)
            D3aaa,D3bbb,D3aab,D3abb = set_denoms.D3denomFast(epsaa,epsbb,occ_aa,occ_bb,virt_aa,virt_bb,n)
            self.denomInfo.update({"D1aa":D1aa,"D1bb":D1bb,"D3aaa":D3aaa,"D3bbb":D3bbb,"D3aab":D3aab,"D3abb":D3abb})

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
        print(type(orb),np.shape(orb))
        orb = np.array(orb)
        print(type(orb),np.shape(orb))
        coeff_aa = orb[0][:,self.dropcore:]
        coeff_bb = orb[1][:,self.dropcore:]
        orb = np.array([coeff_aa,coeff_bb])


        print(np.shape(orb),'final')
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
        print('type',type(pyscf_mf.mo_energy),np.shape(pyscf_mf.mo_energy))

        moE_aa = pyscf_mf.mo_energy[0][self.dropcore:]
        moE_bb = pyscf_mf.mo_energy[1][self.dropcore:]
        self.eps = {"eps_aa":moE_aa,"eps_bb":moE_bb}

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

#        F = np.block([
#                 [      pyscf_mf.get_fock()           ,   np.zeros_like(pyscf_mf.get_fock()) ],
#                 [np.zeros_like(pyscf_mf.get_fock())  ,          pyscf_mf.get_fock()         ]
#                ])
#        fock= C.T@(F@C)
        #print('fock',fock)
        #sys.exit()
        self.integralInfo={"oei":fock,"tei":gmo}



    def get_occInfo(self,pyscf_mf,calcType=None):
        dropcore=self.dropcore
        print('dropcore:',dropcore)
        print('calctype:',calcType)
        if calcType=="fastSIcalc": # running UHF, spin-integrated calc
            print('inside get_occ info:')
            na, nb = pyscf_mf.nelec
            na = na -self.dropcore
            nb = nb-self.dropcore
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
        self.pcc_amps={}

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
 
        if "p" in self.cc_type:
            build_pCC_corrections.drive_pcc_energyCorrections(self)

    def drive_rdm(self,pyscf_mol,pyscf_mf):
        spin_rdm1 = rdm1.build_Spinrdm1(self)
        alpha_rdm,beta_rdm = rdm1.spin_to_spatial_rdm1(self,spin_rdm1)
        # get first-order props
        self.rdm1.update({"spin_rdm":spin_rdm1,"alpha":alpha_rdm,"beta":beta_rdm})   
        props.dipole_moment(pyscf_mol,pyscf_mf,alpha_rdm+beta_rdm,spin_rdm1)
  



class RunXacc(SetupCC):
    """
    The `Run_xacc` class handles perturbative corrections - based on MBPT - given a set of converged,
    infinite-order unitary coupled cluster amplitudes. Currently, the class simply computes every 
    correction we have derived: [4S] and [6S] w.r.t UCCD, [T] w.r.t UCCSD, as well as corrections thru
    fourth-order w.r.t. pUCCD. Future work incorporate quadruples' corrections as well. 
    Subclass of SetupCC(), in case it is necessary to use some of its' methods to adapt to 
    spatial orbital demands. The class is initialized from data read from input files 
    that are a product of a prior UCC calculation perfromed in XACC. 
    (e.g. background information, T-amplitudes, and two-electron integrals) 
    The purpose of the class is to compute the previously derived energy corrections.

    Attributes:
        nocc (int): Number of occupied orbitals in the system.
        nvirt (int): Number of virtual orbitals in the system.
        mo_energies (list of float): Molecular orbital energies for both alpha and beta spins.
        o (slice): Slice object representing the occupied orbitals.
        v (slice): Slice object representing the virtual orbitals.
        denoms (dict): Dictionary storing the denominators for the CC calculation.
        t2amps (ndarray): 4D numpy array of T2 amplitudes for CC calculations.
        t1amps (ndarray): 2D numpy array of T1 amplitudes for CC calculations.
        tei (ndarray): 4D numpy array of two-electron integrals.

    Methods:
        __init__(bkgrd_infile, tamp_infile=None, tei_infile=None):
            Initializes the `run_xacc` object by reading background info printed by xacc, T-amplitudes, 
            and two-electron integrals from input files.
        
        set_denoms():
            Computes the denominators for the CC calculation based on molecular orbital energies.

        ccd_energyTest():
            Calculates and prints a test energy for CCD.

        mp2_energy():
            Calculates the MP2 correlation energy and prints the result, given a set of read in integrals/amps.

        read_tei(tei_infile):
            Reads two-electron integrals from a specified input file.

        read_tamps(tamp_infile):
            Reads the T1 and T2 CC amplitudes from a specified input file.

        read_bkgrd(bkgrd_infile):
            Reads background information printed by xacc (number of occupied/virtual orbitals and MO energies).
    """
    
    def __init__(self,CCbase='pUCCD',bkgrd_infile=None,tamp_infile=None,tei_infile=None,ref='spin-orbital',pyscf_mf=None,pyscf_mol=None,cc_runtype=None):
        """
        Initializes the `run_xacc` object by reading background information from the `bkgrd_infile`, 
        CC amplitudes from the `tamp_infile`, and two-electron integrals from the `tei_infile`. 
        Sets up the necessary data structures for subsequent MBPT correction.
        TO DO:::::: Need to add support specifying which orbitals I am reading in/working with? 
        Spin-orbitals or spatial orbitals? Idea: What if I add 'run_xacc' as a subclass to the driveCC
        SCF handler -- this way I can reuse some of the functionals to build the MO denoms, etc, if 
        necessary

        :param bkgrd_infile: Path to the background input file containing the number of occupied and 
                              virtual orbitals and molecular orbital energies.
        :param tamp_infile: (Optional) Path to the input file containing T1 and T2 CC amplitudes. 
                             Default is None.
        :param tei_infile: (Optional) Path to the input file containing the two-electron integrals. 
                           Default is None.

        :param ref: The reference frame in which we are working; a choice between the 'spin-orbital' or
                    'spatial' representation. If working in 'spatial' frame, will need to convert all
                    tensors.
                    Default is 'spin-orbital'
        """
        self.nocc=None
        self.nvirt=None
        self.mo_energies=None
        self.oei=None 
        self.read_bkgrd(bkgrd_infile,ref)

        self.o=slice(None,self.nocc)
        self.v=slice(self.nocc,None)
        self.t2amps=np.zeros((self.nvirt,self.nvirt,self.nocc,self.nocc))
        self.t1amps=np.zeros((self.nvirt,self.nocc))
        self.read_tamps(tamp_infile,ref)
        if ref == 'spin-orbital':
            self.denomInfo={}
            self.set_denoms(ref,self.o,self.v)
    
            nbas=self.nocc+self.nvirt
            self.tei=np.zeros((nbas,nbas,nbas,nbas))
            self.read_tei(tei_infile)
            #self.mp2_energy()
            #sys.exit()
        elif ref == "spatial":
            # call constructor to inherit class' methods; useful in the case of
            # spatial orbital methods
            SetupCC.__init__(self,pyscf_mf,pyscf_mol,cc_runtype)
            self.tamps={}
            self.convert_t2_spatial(self.t2amps)
            self.pcc_amps ={}
            print('mo energies:',self.eps["eps_aa"])
            build_pCC_corrections.drive_pcc_energyCorrections(self)

        #self.ccd_energyTest()
        #self.mp2_energy()

    def set_denoms(self,ref,o,v):
        eps_a = np.asarray(self.mo_energies)
        eps_b = np.asarray(self.mo_energies)
        n=np.newaxis

        if ref == 'spatial':
            self.D1_aa,self.D1_bb=set_denoms.D1denomFast(eps_a,eps_b,o,o,v,v,n)
            self.D2_aa,self.D2_bb,self.D2_ab=set_denoms.D2denomFast(eps_a,eps_b,o,o,v,v,n)
        else:
            eps = np.append(eps_a, eps_b)
            eps=np.sort(eps)
            self.denomInfo.update({'D1aa':  set_denoms.D1denomSlow(eps,o,v,n)})
            self.denomInfo.update({'D2aa':set_denoms.D2denomSlow(eps,o,v,n)})        
            self.denomInfo.update({'D3aa':set_denoms.D3denomSlow(eps,o,v,n)})
        
    def ccd_energyTest(self):
        """
        Performs a test energy calculation for the CCD method using T2 amplitudes and two-electron integrals. 
        This is primarily for debugging or testing purposes to ensure correct CC energy calculations.

        :return: None
        """
        n=np.newaxis
        o=slice(None,self.nocc)
        v=slice(self.nocc,None)
        ccd_test=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v],self.t2amps)
        print('ccd test E:',ccd_test)

    def mp2_energy(self):
        n=np.newaxis
        o=slice(None,self.nocc)
        v=slice(self.nocc,None)
        print(self.mo_energies,type(self.mo_energies[0]),self.nocc,o,v)
        eps_a = np.asarray(self.mo_energies)
        eps_b = np.asarray(self.mo_energies)
        eps = np.append(eps_a, eps_b)
        eps=np.sort(eps)
        e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    
        t2=e_abij*self.tei[v,v,o,o]
    
        mp2E=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v], t2)

        ccd_test=0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v],self.t2amps)
        print('ccd test E:',ccd_test)
        print('i',self.nocc,'a',self.nvirt)
        print('mp2E: w/ pUCCD amps', mp2E)

        print('final check:',0.250000000000000 * np.einsum('jiab,abji',self.tei[o, o, v, v], self.t2amps))

        import pycc.pcc_base as pcc_base
        print(pcc_base.get_WnT2_energy(self.t2amps,self.tei[v,v,o,o]))

    def read_tei(self,tei_infile):
        """
        Reads two-electron integrals from the input file and stores them in a 4D numpy array.
        accessible as a class attribute

        :param tei_infile: Path to the input file containing two-electron integrals.

        :return: None
        """

        tei={}
        with open(tei_infile,'r') as f:
            for line in f:
                tei_key=float(line.split()[-1])
                index_list=line.split()
                operator_list=[]
                for operator in range(4): # max T2, min T1
                    if index_list[operator] == '|':
                        break
                    operator_list.append(int(index_list[operator].strip('^')))
                #print('op list/key:',operator_list,tei_key,operator_list[0])
                idx=str(operator_list[0])+','+str(operator_list[1])+','+str(operator_list[2])+','+str(operator_list[3])
                a=operator_list[0]
                b=operator_list[1]
                c=operator_list[2]
                d=operator_list[3]
                self.tei[a,b,d,c]=4.0*tei_key

                tei.update({tei_key:operator_list})
        #self.tei=expand_tei(tei,self.nocc,self.nvirt)
        #print('tei',self.tei)

    def convert_tei_2spatial(self,tei,nbas_spinorbs):
        from collections import Counter
        nbas_spat = int(nbas_spinorbs/2)
        W_aaaa = W_bbbb = W_abab = np.zeros((nbas_spat,nbas_spat,nbas_spat,nbas_spat))
        icount=jcount=kcount=lcount=0
        for i in range(nbas_spinorbs):
            for j in range(nbas_spinorbs):
                for k in range(nbas_spinorbs):
                    for l in range(nbas_spinorbs):
                        counts = Counter([i, j, k, l])

                        # Check if any value occurs 3 or more times
                        if any(count >= 3 for count in counts.values()) or i==j or k==l:
                            pass # Go to next cycle, because the value of the 2e-int == 0

                        if i%2 ==0 and j%2==0 and k%2==0 and l%2==0: #aaaa type
                            W_aaaa[i//2,j//2,k//2,l//2]=4.0*tei[i,j,k,l]
                        elif i%2==1 and j%2==1 and k%2==1 and l%2==1: #bbbb type
                            W_bbbb[i//2,j//2,k//2,l//2]=tei[i,j,k,l]
                        elif (                                        #abab type
    (i % 2 == 0 and j % 2 == 1 and k % 2 == 0 and l % 2 == 1) or
    (i % 2 == 1 and j % 2 == 0 and k % 2 == 1 and l % 2 == 0) or
    (i % 2 == 1 and j % 2 == 0 and k % 2 == 0 and l % 2 == 1) or
    (i % 2 == 0 and j % 2 == 1 and k % 2 == 1 and l % 2 == 0)
):
                            W_abab[i//2,j//2,k//2,l//2]=tei[i,j,k,l]
        return W_aaaa,W_bbbb,W_abab

    def read_tamps(self,tamp_infile,ref):
        """
        Reads T1 and T2 CC amplitudes from the specified input file.
        and stores result in class attribute. 

        :param tamp_infile: Path to the input file containing T1 and T2 amplitudes.

        :return: None
        """
        t2amp={}
        t1amp={}
        read_amps=False
        with open(tamp_infile,'r') as f:
            for line in f:
                if read_amps:#have to uncomment 109,13 for nospace
                    #amp_key=float(line.split()[-1])#.strip('|'))
                    amp_key=float(line.split()[-1].strip('|'))
                    #index_list=line.split()
                    index_list=line.split()[:-1]
                    index_list.append('|')
                    index_list.append(amp_key)
                    operator_list=[]
                    for operator in range(4): # max T2, min T1
                        if index_list[operator] == '|':
                            break
                        operator_list.append(int(index_list[operator].strip('^')))
                    #print('op list:',operator_list,'amp key:',amp_key)
                    if len(operator_list)==4: #dealing with t2amp
#                        t2amp.update({amp_key:operator_list})
                        a=operator_list[0]-self.nocc
                        b=operator_list[1]-self.nocc
                        i=operator_list[2]
                        j=operator_list[3]
                        self.t2amps[a,b,i,j]=amp_key
                        self.t2amps[b,a,i,j]= -1.0* amp_key
                        self.t2amps[a,b,j,i]= -1.0*amp_key
                        self.t2amps[b,a,j,i]=amp_key
                    else: # dealing with t1amp
                        a=operator_list[0]-self.nocc
                        i=operator_list[1]
                        self.t1amps[a,i]=amp_key

                if line[:5]=="+++++":#parse the file until this str is read
                    read_amps=True

        #print('t1:',self.t1amps)
        #print('t2:',self.t2amps)
        self.t2amps=self.t2amps.transpose(2,3,1,0)# ijab -> ijba convention ZWW 1/16/25
        self.t1amps=self.t1amps.transpose(1,0)

    def convert_t2_spatial(self,t2_spin):
        nvirt_spin=np.shape(t2_spin)[0]
        nocc_spin=np.shape(t2_spin)[3]

        nvirt_spat=int(nvirt_spin/2)
        nocc_spat=int(nocc_spin/2)
        t2_spat=np.zeros((nocc_spat,nocc_spat,nvirt_spat,nvirt_spat))
        t2_aa= t2_bb=t2_spat
        counter=0
        for a in range(0,nvirt_spin,2):
            for i in range(0,nocc_spin,2):
                a_spat=int(a//2)
                i_spat=int(i//2)
                # -1.0* prefactor necessary to ensure conventions bt xacc/pycc
                t2_spat[i_spat,i_spat,a_spat,a_spat]=(-1.0)**(counter)*t2_spin[a,a+1,i,i+1]
                counter+=1

        self.tamps.update({"t2aa":t2_aa,"t2bb":t2_bb,"t2ab":t2_spat})


    def read_bkgrd(self,bkgrd_infile,ref):
        """
        Reads pertinent background information, such as the number of occupied and virtual orbitals,
        as well as the molecular orbital energies from the specified input file.

        :param bkgrd_infile: Path to the background input file.

        :param ref: specifies whether or not we are pursuing corrections w.r.t. 'spatial' or
                    'spin-orbital' (s)
        :return: None
        """
        with open(bkgrd_infile,'r') as f:
            lines=f.readlines()
        self.nocc=2*int(lines[1].strip().split()[-1])
        self.nvirt=2*int(lines[2].strip().split()[-1])
        tmp_energies=lines[3].strip().split()[-1]
        mo_energies=[]
        for element in tmp_energies.split(','):
            mo_energies.append(float(element.strip('[').strip(']')))
     
        # Add mo energy info to class variable storing oei 
        self.mo_energies=mo_energies
        eps = np.append(mo_energies,mo_energies)
        eps = np.sort(eps)
        self.oei=np.diag(eps)

#        if ref == "spatial": # defines nocc/nvirt w.r.t. # spatial orbs
#            self.nocc = self.nocc - int(lines[1].strip().split()[-1])
#            self.nvirt = self.nvirt - int(lines[2].strip().split()[-1])


class XaccCorrection(RunXacc):
    def __init__(self,*args,**kwargs):
        RunXacc.__init__(self,*args, **kwargs)
        self.denoms=self.denomInfo
        D2 = self.denoms["D2aa"]#.transpose(2,3,0,1)
        D3 = self.denoms["D3aa"]
        D1 = self.denoms["D1aa"]
        o=self.o
        v=self.v
        nocc=self.nocc
        nvirt=self.nvirt
        W = self.tei
        F = self.oei
        T1 = self.t1amps
        T2 = self.t2amps
        self.t2amps_all = {}
        self.pccE_correction ={}
        if 'pUCCD' in args:
            # Build second-order energy correction, and 1st & 2nd-order wavefxn correction
            self.get_SO_energy(T2,W,D2,o,v,self.t2amps_all,self.pccE_correction)
            self.build_t2_SO(T2,W,D2,o,v,self.t2amps_all)
           
            # Use 2nd-order wavefxn to construct 3rd order energy correction, and construct
            self.get_TO_energy(W,D2,o,v,self.t2amps_all,self.pccE_correction)



            # Construct the portion of the 3rd-order wavefxn associated with pure doubles-only, then calculate
            # <0|WnT2'''|0> and 0.25*<0|(T2'^)^2 WnT2'|0>, where 
            #                       T2''' represents the 3rd order pure doubles wavefxn and
            #                       T2' represents the 1st order wavefxn
            self.build_t2_TO(T2,W,D2,o,v,self.t2amps_all)
            self.get_FO_linear_energy(T2,T2.transpose(2,3,0,1),W,D1,D2,D3,o,v,self.t2amps_all,self.pccE_correction)
            self.get_newOverlap(T2,o,v,self.t2amps_all,self.pccE_correction)
            self.get_FO_quadratic_energy(T2,D2,W,o,v,self.t2amps_all,self.pccE_correction)


            self.finalize('pUCCD',self.pccE_correction)
            print('Shutting down....')
            sys.exit()
        elif 'T+' in args:
            # Calculate 4th order [T] first
            triples_E4,t3_SO = self.get_FO_triples(W,T2,o,v,D3,self.pccE_correction,self.t2amps_all)

            # Now get 5th order triples corrections
            triples_E5,t3_TO = self.get_FIFTHO_triples(W,T1,T2,t3_SO,D3,D2,o,v,self.pccE_correction,self.t2amps_all)

            # Get 6th order pure triples

            # Get 6th order pure quadruples
            #self.get_SIXTHO_QUADS(T2,T3,D2,D3,W,o,v)
            # Get 6th order mixed T3/T4

    def get_SO_energy(self,T2,W,D2,o,v,t2amps_all,pccE_correction):
        fullMP2_base =  W[o,o,v,v] * D2
        tmpfullMP2_base = np.copy(fullMP2_base)
        odMP2_base = pcc_base.kill_Diag_T2(tmpfullMP2_base,self.nocc,self.nvirt)
        t2FO_full = T2 + odMP2_base
        self.t2amps_all.update({"mp2_full":fullMP2_base,
                                "mp2_od":odMP2_base,    
                                "t2FO_full":T2+odMP2_base})

        fullMP2_E = pcc_base.get_WnT2_energy(fullMP2_base,self.tei[v,v,o,o])
        odMP2_E   = pcc_base.get_WnT2_energy(odMP2_base,self.tei[v,v,o,o])
        pccE_correction.update({"mp2_full":fullMP2_E,"mp2_od":odMP2_E,"Total E(2) from doubles:":odMP2_E})

        return 

    def build_t2_SO(self,T2,W,D2,o,v,t2amps_all):
        # build Q2od RV T2pucc
        SO_base = pcc_base.build_LCCD_T2(T2,W,o,v,D2)
        odSO_base = pcc_base.kill_Diag_T2(np.copy(SO_base),self.nocc,self.nvirt)

        # now build (Q2all RV (Q2od RVod)), and partition
        t2_FO_od = t2amps_all["mp2_od"]
        mp_base = pcc_base.build_LCCD_T2(t2_FO_od,W,o,v,D2)
        mp_base_diag = pcc_base.return_Diag_T2(np.copy(mp_base),self.nocc,self.nvirt)
        mp_base_od   = pcc_base.kill_Diag_T2(np.copy(mp_base),self.nocc,self.nvirt)

        # construct the full SO correction
        t2_SO_full = odSO_base + mp_base


        print('odso ovrlap:',pcc_base.get_WnT2_energy(odSO_base,odSO_base.transpose(2,3,0,1)))
        print('t2SO_mp3 ovrlap:',pcc_base.get_WnT2_energy(mp_base,mp_base.transpose(2,3,0,1)))#t2amps_all["t2SO_mp3"],t2amps_all["t2SO_mp3"].transpose(2,3,0,1)))
        # now build full MP3 base as bkup 
        fullMP2_base = t2amps_all["mp2_full"]
        fullMP3_base = pcc_base.build_LCCD_T2(fullMP2_base,W,o,v,D2)

        print(pcc_base.get_WnT2_energy(t2_SO_full,t2_SO_full.transpose(2,3,0,1)))
        t2amps_all.update({"t2SO_pucc":odSO_base,
                          "t2SO_mp3":mp_base,
                          #"t2SO_mp3":t2_SO_full,
                          "t2SO_full":t2_SO_full,
                          "t2SO_mp3base":fullMP3_base,
                          "t2SO_mp_diag":mp_base_diag,
                          "t2SO_mp_od":mp_base_od})

        test = t2amps_all["t2SO_mp3"]
        print('testing t2SO_mp3 ovrlap:',pcc_base.get_WnT2_energy(test,test.transpose(2,3,0,1)))
        return

    def get_TO_energy(self,W,D2,o,v,t2amps_all,pccE_correction):
        # get off-diagonal MP3
        t2SO_mpOD = t2amps_all["t2SO_mp_od"]
        mp3_od_E = pcc_base.get_WnT2_energy(t2SO_mpOD,W[v,v,o,o])
        print('testE;',mp3_od_E)

        # get Q2od RV T2od
        t2SO_pucc = t2amps_all["t2SO_pucc"]
        pucc_TO_E = 2.0*pcc_base.get_WnT2_energy(t2SO_pucc,W[v,v,o,o])
        print('pucc_TO_E',pucc_TO_E)
        
        # test 
        test = t2amps_all["t2SO_full"]
        print('testing (3):',pcc_base.get_WnT2_energy(test,W[v,v,o,o]))
        # total 3rd order contrib
        totalE3 = pucc_TO_E + mp3_od_E


        t1 = t2amps_all["t2SO_mp_od"]
        t2 = t2amps_all["t2SO_mp_diag"]
        t3 = t2amps_all['t2SO_pucc']
        t_e = pcc_base.get_WnT2_energy(t1,W[v,v,o,o]) + pcc_base.get_WnT2_energy(t2,W[v,v,o,o]) + pcc_base.get_WnT2_energy(t3,W[v,v,o,o])
        print('t_e',t_e)
        print(pcc_base.get_WnT2_energy(t1,W[v,v,o,o]),pcc_base.get_WnT2_energy(t2,W[v,v,o,o]),pcc_base.get_WnT2_energy(t3,W[v,v,o,o]))
        #sys.exit()

        #totalE3 = t_e
        # now get full MP3 energy:
        fullMP3_base = t2amps_all["t2SO_mp3base"]
        fullMP3_E = pcc_base.get_WnT2_energy(fullMP3_base,W[v,v,o,o])
        print('full/od MP3_E:', fullMP3_E,mp3_od_E)
        #sys.exit()

        #save this info
        pccE_correction.update({"mp3_full":fullMP3_E,"mp3_TO_od":mp3_od_E,
            "pucc_TO":pucc_TO_E, "Total E(3) from doubles:":totalE3})

        
        print("Total E(3) from doubles:",totalE3)



    def build_t2_TO(self,T2,W,D2,o,v,t2amps_all):
        # first, build MP4-like ladders from second-order wavefxn "t2SO_full"
        t2SO_full = t2amps_all["t2SO_full"]
        t2_TOorig = pcc_base.build_LCCD_T2(np.copy(t2SO_full),W,o,v,D2)

        # have to add Q2od RV Q2d RV T2puccd
        tmp = pcc_base.build_LCCD_T2(T2,W,o,v,D2)
        tmp = pcc_base.return_Diag_T2(tmp, self.nocc,self.nvirt)
        tmp = pcc_base.build_LCCD_T2(tmp,W,o,v,D2)
        tmp = pcc_base.kill_Diag_T2(tmp,self.nocc,self.nvirt)
        print('t2SO_full:',pcc_base.get_WnT2_energy(tmp,tmp.transpose(2,3,0,1)),
                pcc_base.get_WnT2_energy(t2SO_full,t2SO_full.transpose(2,3,0,1))) 
        t2_TO_mp4ladder = t2_TOorig + tmp
        t2amps_all.update({"t2TO_mp4":t2_TO_mp4ladder})
        print('MP4 LADDER ENERGY TEST test:',pcc_base.get_WnT2_energy(t2_TO_mp4ladder,W[v,v,o,o]))
        tmp = t2amps_all["t2SO_mp3"]
        print('final t2SO_mp3 ovrlap:',pcc_base.get_WnT2_energy(tmp,tmp.transpose(2,3,0,1)))

        # now have to build 0.5*T2^WT2 using first-order T2
        t2_FO_full = t2amps_all["t2FO_full"]
        quad_term1 = 0.5*ucc_eqns.uccsd_T2dagWnT2(W,t2_FO_full,o,v)
        T2dag = T2.transpose(2,3,0,1)
        W_diag = pcc_base.return_Diag_T2(np.copy(W[o,o,v,v]), self.nocc,self.nvirt)
        roovv = -0.250000000 * np.einsum("ikab,cdkl,jlcd->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += 0.062500000 * np.einsum("klab,cdkl,ijcd->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += -0.250000000 * np.einsum("ijac,cdkl,klbd->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += 1.000000000 * np.einsum("ikac,cdkl,jlbd->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += -0.250000000 * np.einsum("klac,cdkl,ijbd->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += 0.062500000 * np.einsum("ijcd,cdkl,klab->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv += -0.250000000 * np.einsum("ikcd,cdkl,jlab->ijab",T2,T2dag,W_diag,optimize="optimal")
        roovv = 0.5*roovv
        roovv = pcc_base.return_Diag_T2(np.copy(roovv),self.nocc,self.nvirt)
        quad_term1 = quad_term1 - roovv
        quad_term1 = tamps.antisym_T2(quad_term1,None,None)
        quad_term1 = quad_term1 *D2
        t2amps_all.update({"t2TO_t2dagwt2":quad_term1})

        # now have to build 0.5*WT2^2
        quad_term2 = 0.5*ucc_eqns.uccsd_wnT2sqr(W,t2_FO_full,o,v)
        W_diag = W_diag.transpose(2,3,0,1)
        quad_term2tmp = -0.250000000 * np.einsum("ikab,jlcd,cdkl->ijab",T2,T2,W_diag,optimize="optimal")
        quad_term2tmp += 0.062500000 * np.einsum("klab,ijcd,cdkl->ijab",T2,T2,W_diag,optimize="optimal")
        quad_term2tmp += -0.250000000 * np.einsum("ijac,klbd,cdkl->ijab",T2,T2,W_diag,optimize="optimal")
        quad_term2tmp += 0.500000000 * np.einsum("ikac,jlbd,cdkl->ijab",T2,T2,W_diag,optimize="optimal")
        quad_term2tmp =  pcc_base.return_Diag_T2(np.copy(quad_term2tmp),self.nocc,self.nvirt)
        quad_term2  -= 0.5*quad_term2tmp

        quad_term2 = tamps.antisym_T2(quad_term2,None,None) 
        quad_term2 = quad_term2 *D2
        t2amps_all.update({"t2TO_wt2sqr":quad_term2})

        t2amps_all.update({"t2TO_full":quad_term1+quad_term2+t2_TO_mp4ladder})


    def get_FO_linear_energy(self,T2,T2dag,W,D1,D2,D3,o,v,t2amps_all,pccE_correction):
        t2TO_full = t2amps_all["t2TO_full"]
        doublesE = pcc_base.get_WnT2_energy(t2TO_full,W[v,v,o,o])
        print('Doubles-only contrib to WnT2 @ E(4):',doublesE)
        # now build [S]/[T]
        t2FO_full = t2amps_all["t2FO_full"]
        singles_E4,t1_SO = self.get_FO_singles(W,t2FO_full,o,v,D1,pccE_correction)
        triples_E4,t3_SO = self.get_FO_triples(W,t2FO_full,o,v,D3,pccE_correction)
        print('[S] and [T]',singles_E4,triples_E4)
        
        t2amps_all.update({"t1":t1_SO,"t3":t3_SO})

        #finally, update the 3rd order T2 w/ T1/T3 info in preparation for overlap
        # calculation
        t2_TO_trips = self.build_sqrBrakT_T2(t2FO_full,W,o,v,D2,D3)
        t2_TO_sing = self.build_sqrBrakS_T2(t2FO_full,W,o,v,D2,D1)
        t2_TO_all = t2TO_full +t2_TO_trips + t2_TO_sing
        t2amps_all.update({"t2TO_full":t2_TO_all})
        
        totalFO_E = singles_E4 + triples_E4 + doublesE #+ pccE_correction["-0.25*(T2^)^2WT2"]
        pccE_correction.update({"Total E(4) from doubles:":doublesE,
            "Total E(4) w/ [S], [T]:":totalFO_E})


    def get_newOverlap(self,T2,o,v,t2amps_all,pccE_correction):
        t2FO_full = t2amps_all["t2FO_full"]
        t2SO_full = t2amps_all["t2SO_full"]
        t2TO_full = t2amps_all["t2TO_full"]

        t1SO = t2amps_all["t1"]
        t3SO = t2amps_all["t3"]
        ##### TO DO: NEED TO ADD SO T1 AND T3 INTO THE CALCULATION OF THE OVERLAP
        ####  ZWW 1/2025
        storage =[pcc_base.get_WnT2_energy(t2FO_full,t2FO_full.transpose(2,3,0,1)),
                    pcc_base.get_WnT2_energy(t2SO_full,t2SO_full.transpose(2,3,0,1)),
                    2.0*pcc_base.get_WnT2_energy(t2FO_full,t2SO_full.transpose(2,3,0,1)),
                    2.0*pcc_base.get_WnT2_energy(t2FO_full,t2TO_full.transpose(2,3,0,1))]
        overlap = ( pcc_base.get_WnT2_energy(t2FO_full,t2FO_full.transpose(2,3,0,1))
                   + pcc_base.get_WnT2_energy(t2SO_full,t2SO_full.transpose(2,3,0,1))
                   + 2.0*pcc_base.get_WnT2_energy(t2FO_full,t2SO_full.transpose(2,3,0,1))
                   + 2.0*pcc_base.get_WnT2_energy(t2FO_full,t2TO_full.transpose(2,3,0,1))
                   + 1.0 )
        print('overlap:',overlap)
        for i in range(len(storage)):
            print('overlap element ',i,': ',storage[i])
        pccE_correction.update({"Total Overlap":overlap})

    def get_FO_quadratic_energy(self,T2,D2,W,o,v,t2amps_all,pccE_correction):
        t2dagwnt2 = t2amps_all["t2TO_t2dagwt2"]
        quad_E = -0.5*pcc_base.get_WnT2_energy(t2dagwnt2,W[v,v,o,o])
        print('quadE ', quad_E)


        t2FO = t2amps_all["t2FO_full"]
        tmp = t2dagwnt2 / D2
        quad_E = -0.5*pcc_base.get_WnT2_energy(tmp,t2FO.transpose(2,3,0,1))
        print(quad_E,'new quad E **** here')
        pccE_correction.update({"-0.25*(T2^)^2WT2":quad_E})
        #sys.exit()

        totalFO_E = pccE_correction["Total E(4) w/ [S], [T]:"] + quad_E
        pccE_correction.update({
            "Total E(4) w/ [S], [T], and -0.25*(t2^)^2wt2:":totalFO_E})


    def get_FO_triples(self,W,T2,o,v,D3,pccE_correction):
        D3T3 = build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,T2)
        D3T3 = tamps.antisym_T3(D3T3,None,None)
        T3 = D3T3*D3
        sqrBrak_T =0.25* build_sqrbrak_corrections.sqr_brakT_spin(D3T3,T3.transpose(3,4,5,0,1,2))
        print('[T] correction to pUCCD:',sqrBrak_T)
        pccE_correction.update({"Triples' [T]":sqrBrak_T})
        return sqrBrak_T, T3


    def get_FIFTHO_triples(self,W,T1,T2,T3SO,D3,D2,o,v,pccE_correction,t2amps_all):
        # build Q3 [W,T3SO]
        D3T3 = build_sqrbrak_corrections.buildTO_WT3_to_T3(W,o,v,T3SO)
        D3T3 = tamps.antisym_T3(D3T3,None,None)
        E5_t3SOdag_wnT3SO = 0.25* build_sqrbrak_corrections.sqr_brakT_spin(T3SO,D3T3.transpose(3,4,5,0,1,2))
        print('E(5) T3SO^ Q3(WnT3SO) :',E5_t3SOdag_wnT3SO)
        T3_TO = D3T3*D3 # store wnT3SO -> T3 object


        D3T3=0.0
        # Build Q3 0.5*[[W,T2],T2] 
        D3T3 = build_sqrbrak_corrections.buildTO_wnT2sqr_to_T3(W,o,v,T2)
        D3T3 = tamps.antisym_T3(D3T3,None,None)
        E5_wnT3SOdag_wnt2sqr = 0.25* build_sqrbrak_corrections.sqr_brakT_spin(T3SO,D3T3.transpose(3,4,5,0,1,2))
        print('E(5) T3SO^ Q3(wnT2^2):',E5_wnT3SOdag_wnt2sqr)
        T3_TO += D3T3*D3 # now I have both Q3 wnT2^2 + WT3SO 
        t2amps_all.update({"T3_TO":,T3_TO})


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
        ## ** need to make a T2_FO from T3_TO above, and the following below....

        # Now build <0|Hbar2 C^[3]|0> for Hbar2 == [V,T2]
        t2_fromT3SO = self.build_sqrBrakT_T2(T2,W,o,v,D2,D3)
        t2tmp = pcc_base.build_LCCD_T2(t2_fromT3SO,W,o,v,D2)
        t2tmp = t2tmp/D2
        E5_hbar2_c3_wt2 = pcc_base.get_WnT2_energy(T2,t2tmp.transpose(2,3,0,1))

        #newd3t3 = build_sqrbrak_corrections.build_T3_secondO_spin(W,o,v,t2tmp)
        #newd3t3 = tamps.antisym_T3(newd3t3,None,None)
        #E5_hbar2dag_c5 = 0.25* build_sqrbrak_corrections.sqr_brakT_spin(T3SO,newd3t3.transpose(3,4,5,0,1,2))
        print('E(5) Hbar2^ C3 from wnT2:',E5_hbar2_c3_wt2)



        triples_E5= E5_wnT3SOdag_wnt2sqr + E5_t3SOdag_wnT3SO+ E5_hbar2_c3_wt2 + hbar2_c3_wnT1_C3
        print('Fifth-order triples correction:',triples_E5)
        t3_TO=T3
        return triples_E5, t3_TO


    def get_SIXTHO_TRIPS(self,T2,T3SO,D2,D3,W,o,v):
        # Q2 .. Q3.. [[W,T1],T2
        D3T3 =0.0
        D3T3 = build_sqrbrak_corrections.buildTO_wnT1T2_to_T3(W,o,v,T1,T2)
        D3T3 = tamps.antisym_T3(D3T3,None,None)
        T3 = D3T3*D3

        E5_wnT3SOdag_wnt1t2 = 0.25* build_sqrbrak_corrections.sqr_brakT_spin(T3SO,D3T3.transpose(3,4,5,0,1,2))
        print('E(6) T3SO^ Q3 (WT1T2):',E5_wnT3SOdag_wnt1t2)


        # Finall build <0|Hbar2 C^[3]|0> + h.c. for Hbar2 == [V,T1]
        d2t2_tmp = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
        d2t2_tmp += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
        d2t2_tmp = tamps.antisym_T2(d2t2_tmp,None,None)


        t2_fromT3 = self.build_sqrBrakT_T2(T2,W,o,v,D2,D3)
        E_hbar2_c3_wnT1_C3 =pcc_base.get_WnT2_energy(t2_fromT3,d2t2_tmp.transpose(2,3,0,1))
        print('E(6) Hbar2^ C3  from wnT1:',hbar2_c3_wnT1_C3)


    def get_SIXTHO_QUADS(self,T2,T3,D2,D3,D4,W,o,v):
        D4T4_wnt2sqr = build_sqrbrak_corrections.build_SIXTHOQUADS_wnt2sqr(T2,W,o,v)
        D4T4_wnt3 = build_sqrbrak_corrections.build_SIXTHOQUADS_wnt3(T3,W,o,v)

    

    def get_FO_singles(self,W,T2,o,v,D1,pccE_correction):
         D1T1 = build_sqrbrak_corrections.build_T1_fromT2_SOspin(W,o,v,T2)
         T1approx = D1T1*D1
         sqrBrak_S = np.einsum("ia,ai->",D1T1,T1approx.transpose(1,0))
         print('[S] correction to pUCCD', sqrBrak_S)
         pccE_correction.update({"Singles' [S]":sqrBrak_S})
         return sqrBrak_S, T1approx

    def build_sqrBrakS_T2(self,T2,W,o,v,D2,D1):
        totalT2 = T2 #+  pcc_base.kill_Diag_T2(W[o,o,v,v]*D2,self.nocc,self.nvirt)
        D1T1 = build_sqrbrak_corrections.build_T1_fromT2_SOspin(W,o,v,totalT2)
        T1 = D1T1*D1
        roovv = -0.500000000 * np.einsum("ka,ijkb->ijab",T1,W[o,o,o,v],optimize="optimal")
        roovv += -0.500000000 * np.einsum("ic,jcab->ijab",T1,W[o,v,v,v],optimize="optimal")
        roovv = tamps.antisym_T2(roovv,None,None)
        roovv = roovv*D2
        return roovv

    def build_sqrBrakT_T2(self,T2,W,o,v,D2,D3):
        import pycc.wicked_T3corr as wicked_T3corr
        t3resid=wicked_T3corr.build_T3_secondO(W,o,v,T2)
        t3resid=tamps.antisym_T3(t3resid,None,None)
        t3residOrigContract=t3resid
        #t3resid=t3resid.transpose(3,4,5,0,1,2)
        t3=t3resid*D3
        t3Contract=t3
        #t3=t3.transpose(3,4,5,0,1,2)

        netT2=wicked_T3corr.build_netT2(W,o,v,t3)
        netT2=wicked_T3corr.antisym_T2(netT2,self.nocc,self.nvirt)
        netT2 = netT2*D2
        return netT2 
        #t2_likeE=0.250000000 * np.einsum("abij,ijab->",T2dag,netT2,optimize="optimal")
        #print('[T] contrib to third-order overlap: ',t2_likeE)

    def finalize(self,label='pUCCD',dataDict={}):
        print('\n\n\n\n\n ')
        print('**********************')
        print('Summary of Xacc correction results:')
        if "pUCCD" in label:
            print('**********************')
            print('*** Printing log summary of all information: ***')
            print('**********************')
            for key, value in dataDict.items():
                print(f"{key}: {value}")

            doubles_correction = dataDict["Total E(2) from doubles:"] + dataDict["Total E(3) from doubles:"] \
                    + dataDict["Total E(4) from doubles:"] #E2_E3 + E4_doubles
            all_correction = doubles_correction + dataDict["Singles' [S]"] + dataDict["Triples' [T]"]

            print('\n\n**********************')
            print('*** Summary of pUCCD correction results: ***')
            print('**********************')
            for key, value in dataDict.items():
                if "Overlap" in key:
                    print(f"{key}: {value}")

            total_overlap = dataDict["Total Overlap"]  
            print()
            print("pUCCD + [D]:",doubles_correction)
            print("pUCCD + [D] + [S] + [T]:",all_correction)
            print('Renormalized (R)-pUCCD + [D]:',doubles_correction/total_overlap)
            print('Renormalized (R)-pUCCD + [D] + [S] + [T]:',all_correction/total_overlap)
            x_data = np.array([1,1,1])
            y_data = np.array([dataDict["Total E(2) from doubles:"],dataDict["Total E(3) from doubles:"],dataDict["Total E(4) w/ [S], [T], and -0.25*(t2^)^2wt2:"]])
            misc.pade_approximant(x_data,y_data,dataDict)
        print('\n\n**********************')
        print('**********************')
#        print('E(2): ', E2)
#        print('E(3): ', E3)
#        print('E(4): ', E4)
#        print('total Doubles contribution:',E2+E3+E4)
#    
#        print('\n\n\n\n\n ')
#        print('**********************')
#        print('Summary of (singles/triples) results:')
#        print('E(4) [S]:',E4_singlesFO)
#        print('E(4) [T]:',E4_triplesFO)
#    
#        print('Total correction to pUCCD thru fourth-order, including [S] and [T]:', E2+E3+E4+E4_singlesFO+E4_triplesFO)
#        print("Total pUCC+E(2)+E(3)+E(4)+[S]+[T] energy: ", driveCCobj.correlationE["totalE"]+E2+E3+E4+E4_singlesFO+E4_triplesFO)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("Running bare pycc.py")
