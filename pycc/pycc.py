import numpy as np
import pycc.set_denoms as set_denoms
import pycc.tamps as tamps
import pycc.cc_energy as cc_energy
import pycc.cc_eqns as cc_eqns
import pycc.rdm1 as rdm1
import pycc.props as props
import pickle

class SetupCC():
    def __init__(self,pyscf_mf,pyscf_mol,cc_info):#Set the defaults up for CC calculation
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
        ## TO DO:: ADD OPTION FOR SPIN-INTEGRATED CC EQNS, AND INTERFACE TO XACC


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




    def get_occInfo(self,pyscf_mf):
        dropcore=self.dropcore
        print('dropcore:',dropcore)
        if 'RHF' in str(type(pyscf_mf)): # running RHF calculation
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
            nsvirt = 2 * (pyscf_mf.get_fock()[0].shape[0] - na)#(norbs - nocc)
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



        if self.diis_size is not None:# only works for spin-orb models rn
            from pycc.diis import DIIS
            self.diis_update=DIIS(self.diis_size, start_iter=self.diis_start_cycle)
            self.old_vec=tamps.get_oldvec(self.tamps,cc_info["slowSOcalc"])



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
        self.rdm1.update({"alpha":alpha_rdm,"beta":beta_rdm})   
        props.dipole_moment(pyscf_mol,pyscf_mf,alpha_rdm+beta_rdm)
  
"""Provide the primary functions."""


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
